from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.artifacts import ArtifactRepository, SCHEMA_VERSION
from api.config import ApiSettings, _bool_env
from api.database import ApiDatabase
from api.models import (
    AccessDecisionRequest,
    AiSettingsRequest,
    CafeCreateRequest,
    ConversationCreateRequest,
    LoginRequest,
    ManagerReviewRequest,
    MessageCreateRequest,
    OwnerDecisionRequest,
    ProcessDataRequest,
    RunCreateRequest,
    SignupRequest,
)
from api.services.chat import grounded_answer
from api.services.ai_settings import AiSettingsService
from api.services.runs import RunService
from api.services.uploads import process_browser_files


def _envelope(**payload: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **payload}


def _problem(status: int, code: str, message: str, details: Any = None) -> None:
    detail = {"code": code, "message": message}
    if details is not None:
        detail["details"] = details
    raise HTTPException(status_code=status, detail=detail)


def _public_run(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "cafe_id", "status", "stage", "created_at", "updated_at", "analysis_period",
        "report_state", "findings_count", "error_count",
    )
    return {key: row.get(key) for key in keys}


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    if settings is None and not _bool_env("WADDEHHA_SKIP_DOTENV"):
        # Serving path only (`app = create_app()` below, i.e. uvicorn). Without
        # this the API process has no LLM_PROVIDER/API keys, so a run started
        # from the UI fails in every analyst and the chat agent reports itself
        # unconfigured -- even though the CLI, scheduler and `langgraph dev`
        # all read the same .env happily.
        #
        # Deliberately NOT at module import time, and skipped whenever a caller
        # supplies settings (which the test suite always does): loading .env
        # during pytest collection previously put real provider keys into every
        # test process and turned "offline" tests into live API calls.
        #
        # WADDEHHA_SKIP_DOTENV lets the E2E harness opt out, so browser tests
        # assert against a fixed configuration instead of whatever provider and
        # model the developer happens to have in .env.
        from dotenv import load_dotenv

        load_dotenv()
    configured = settings or ApiSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database = ApiDatabase(configured.api_db)
        artifacts = ArtifactRepository(
            configured.project_root, database,
            include_test_evidence=configured.include_test_evidence,
            checkpoint_db=configured.checkpoint_db,
        )
        if configured.seed_users:
            database.seed_development_users(
                configured.seed_accounts or (), [cafe["id"] for cafe in artifacts.list_cafes()]
            )
        if configured.bootstrap_admin:
            database.ensure_local_admin(
                configured.admin_username,
                configured.admin_password,
                [cafe["id"] for cafe in artifacts.list_cafes()],
            )
        app.state.database = database
        app.state.artifacts = artifacts
        app.state.ai_settings = AiSettingsService(configured.project_root / "db" / "ai_settings.dpapi")
        app.state.run_service = RunService(configured, database, artifacts)
        try:
            yield
        finally:
            app.state.run_service.close()

    app = FastAPI(
        title="Waddehha API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs" if configured.environment == "development" else None,
        openapi_url="/api/openapi.json" if configured.environment == "development" else None,
    )
    app.state.settings = configured
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Request-ID", "Last-Event-ID"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if request.url.path.startswith("/api/admin/ai-settings"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_envelope(
                error={
                    "code": "validation_error",
                    "message": "The request was not valid.",
                    "details": details,
                    "request_id": request.state.request_id,
                }
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            error = dict(exc.detail)
        else:
            error = {
                "code": "not_found" if exc.status_code == 404 else "http_error",
                "message": str(exc.detail),
            }
        error["request_id"] = request.state.request_id
        return JSONResponse(status_code=exc.status_code, content=_envelope(error=error))

    def database(request: Request) -> ApiDatabase:
        return request.app.state.database

    def artifacts(request: Request) -> ArtifactRepository:
        return request.app.state.artifacts

    def ai_settings(request: Request) -> AiSettingsService:
        return request.app.state.ai_settings

    def current_user(
        request: Request,
        session: str | None = Cookie(default=None, alias=configured.cookie_name),
    ) -> dict[str, Any]:
        user = request.app.state.database.user_for_session(session)
        if user is None:
            _problem(401, "authentication_required", "Sign in to continue.")
        return user

    def require_roles(*roles: str):
        def dependency(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
            if user["role"] not in roles:
                _problem(403, "permission_denied", "Your role cannot perform this action.")
            return user

        return dependency

    def accessible_cafe(
        cafe_id: str, user: dict[str, Any], repository: ArtifactRepository
    ) -> dict[str, Any]:
        cafe = repository.get_cafe(cafe_id)
        if cafe is None:
            _problem(404, "cafe_not_found", "Café not found.")
        if user["role"] != "owner" and cafe_id not in user["cafe_ids"]:
            _problem(403, "cafe_access_denied", "You are not assigned to this café.")
        return cafe

    def accessible_run(
        run_id: str, user: dict[str, Any], repository: ArtifactRepository
    ) -> dict[str, Any]:
        run = repository.get_run(run_id)
        if run is None:
            _problem(404, "run_not_found", "Run not found.")
        accessible_cafe(run["cafe_id"], user, repository)
        return run

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return _envelope(status="ok", service="waddehha-api", version="0.1.0")

    @app.post("/api/auth/login")
    def login(payload: LoginRequest, response: Response, db: ApiDatabase = Depends(database)) -> dict[str, Any]:
        result = db.authenticate(payload.email, payload.password, configured.session_ttl_seconds)
        if result is None:
            approval_status = db.account_approval_status(payload.email)
            if approval_status == "pending":
                _problem(403, "approval_pending", "Your access request is waiting for owner approval.")
            if approval_status == "rejected":
                _problem(403, "access_rejected", "Your access request was not approved.")
            _problem(401, "invalid_credentials", "Email or password is incorrect.")
        user, token = result
        response.set_cookie(
            key=configured.cookie_name,
            value=token,
            max_age=configured.session_ttl_seconds,
            httponly=True,
            secure=bool(configured.cookie_secure),
            samesite="strict",
            path="/api",
        )
        return _envelope(user=user)

    @app.post("/api/auth/signup", status_code=202)
    def signup(payload: SignupRequest, db: ApiDatabase = Depends(database)) -> dict[str, Any]:
        request_record = db.create_access_request(
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
            role=payload.requested_role,
        )
        if request_record is None:
            _problem(409, "account_exists", "An account or pending request already uses this email.")
        return _envelope(access_request=request_record)

    @app.post("/api/auth/logout", status_code=204)
    def logout(
        session: str | None = Cookie(default=None, alias=configured.cookie_name),
        db: ApiDatabase = Depends(database),
    ) -> Response:
        db.delete_session(session)
        response = Response(status_code=204)
        response.delete_cookie(
            configured.cookie_name,
            path="/api",
            secure=bool(configured.cookie_secure),
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/api/auth/me")
    def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
        return _envelope(user=user)

    @app.get("/api/capabilities")
    def capabilities(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
        """Deployment switches the UI has to know about to render honestly.

        Without this the human gate would have to show an owner buttons that
        the API then 403s, or hide buttons that actually work.
        """
        del user
        return _envelope(
            owner_self_review=configured.allow_owner_self_review,
            local_file_access=configured.environment == "development",
        )

    @app.get("/api/admin/ai-settings")
    def get_ai_settings(
        user: dict[str, Any] = Depends(require_roles("owner")),
        service: AiSettingsService = Depends(ai_settings),
    ) -> dict[str, Any]:
        del user
        return _envelope(settings=service.public())

    @app.post("/api/admin/ai-settings")
    def save_ai_settings(
        payload: AiSettingsRequest,
        user: dict[str, Any] = Depends(require_roles("owner")),
        service: AiSettingsService = Depends(ai_settings),
    ) -> dict[str, Any]:
        del user
        values = payload.model_dump(exclude={"api_key", "tavily_key", "langsmith_key"})
        values.update(
            api_key=payload.api_key.get_secret_value() if payload.api_key else None,
            tavily_key=payload.tavily_key.get_secret_value() if payload.tavily_key else None,
            langsmith_key=payload.langsmith_key.get_secret_value() if payload.langsmith_key else None,
        )
        try:
            settings = service.save(values)
        except (ValueError, RuntimeError, OSError) as exc:
            _problem(422, "ai_settings_invalid", str(exc))
        return _envelope(settings=settings)

    @app.post("/api/admin/ai-settings/test")
    def test_ai_settings(
        user: dict[str, Any] = Depends(require_roles("owner")),
        service: AiSettingsService = Depends(ai_settings),
    ) -> dict[str, Any]:
        del user
        try:
            result = service.test_connection()
        except ValueError as exc:
            _problem(409, "ai_settings_missing", str(exc))
        return _envelope(result=result)

    @app.delete("/api/admin/ai-settings", status_code=200)
    def clear_ai_settings(
        user: dict[str, Any] = Depends(require_roles("owner")),
        service: AiSettingsService = Depends(ai_settings),
    ) -> dict[str, Any]:
        del user
        return _envelope(settings=service.clear())

    @app.get("/api/admin/access-requests")
    def access_requests(
        status: str = Query(default="pending", pattern="^(pending|approved|rejected)$"),
        user: dict[str, Any] = Depends(require_roles("owner")),
        db: ApiDatabase = Depends(database),
    ) -> dict[str, Any]:
        del user
        return _envelope(items=db.list_access_requests(status))

    @app.post("/api/admin/access-requests/{user_id}/decision")
    def access_request_decision(
        user_id: str,
        payload: AccessDecisionRequest,
        user: dict[str, Any] = Depends(require_roles("owner")),
        db: ApiDatabase = Depends(database),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        request_record = db.decide_access_request(
            user_id,
            payload.decision,
            user["id"],
            [cafe["id"] for cafe in repository.list_cafes()],
        )
        if request_record is None:
            _problem(409, "request_not_pending", "This access request is no longer pending.")
        return _envelope(access_request=request_record)

    @app.get("/api/cafes")
    def list_cafes(
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        cafes = repository.list_cafes()
        if user["role"] != "owner":
            cafes = [cafe for cafe in cafes if cafe["id"] in user["cafe_ids"]]
        return _envelope(items=[repository.public_cafe(cafe) for cafe in cafes])

    @app.post("/api/cafes", status_code=201)
    def create_cafe(
        payload: CafeCreateRequest,
        user: dict[str, Any] = Depends(require_roles("owner")),
        db: ApiDatabase = Depends(database),
    ) -> dict[str, Any]:
        cafe = db.create_cafe(payload.model_dump(), user["id"])
        return _envelope(cafe={**cafe, "branch_id": cafe["id"], "data_status": "not_connected"})

    @app.get("/api/cafes/{cafe_id}")
    def get_cafe(
        cafe_id: str,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        cafe = accessible_cafe(cafe_id, user, repository)
        return _envelope(cafe=repository.public_cafe(cafe))

    @app.get("/api/cafes/{cafe_id}/sources")
    def list_sources(
        cafe_id: str,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        accessible_cafe(cafe_id, user, repository)
        return _envelope(cafe_id=cafe_id, items=repository.sources(cafe_id))

    @app.get("/api/cafes/{cafe_id}/weeks")
    def list_available_weeks(
        cafe_id: str,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        """Weeks the POS data actually covers, plus the one to preselect.

        Lets the UI offer a real choice instead of silently inheriting the
        pipeline's calendar default, which analyses an empty window whenever
        the data stops short of today.
        """
        accessible_cafe(cafe_id, user, repository)
        return _envelope(cafe_id=cafe_id, **repository.available_weeks(cafe_id))

    @app.post("/api/cafes/{cafe_id}/data/process", status_code=202)
    def process_cafe_data(
        cafe_id: str,
        payload: ProcessDataRequest,
        request: Request,
        user: dict[str, Any] = Depends(require_roles("owner", "manager")),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        cafe = accessible_cafe(cafe_id, user, repository)
        data_dir = cafe.get("_data_dir")
        if data_dir is None:
            _problem(409, "data_folder_not_connected", "This café does not have a connected local data folder.")
        upload = process_browser_files(
            project_root=configured.project_root,
            cafe_id=cafe_id,
            data_dir=data_dir,
            files=[item.model_dump() for item in payload.files],
        )
        if not upload["accepted"]:
            _problem(422, "no_supported_files", "None of the selected files matched a registered source.", upload["rejected"])
        run = request.app.state.run_service.start(cafe, payload.target_week, user["id"])
        return _envelope(upload=upload, run=_public_run(run))

    @app.get("/api/cafes/{cafe_id}/data/{source}")
    def list_source_data(
        cafe_id: str,
        source: str,
        limit: int = Query(default=50, ge=1, le=200),
        cursor: str | None = None,
        user: dict[str, Any] = Depends(require_roles("owner", "manager")),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        accessible_cafe(cafe_id, user, repository)
        try:
            page = repository.data_page(cafe_id, source, limit=limit, cursor=cursor)
        except ValueError:
            _problem(400, "invalid_cursor", "The cursor is invalid.")
        if page is None:
            _problem(404, "source_data_not_found", "No processed data is available for this source.")
        items, next_cursor = page
        return _envelope(cafe_id=cafe_id, source=source, items=items, next_cursor=next_cursor)



    @app.get("/api/cafes/{cafe_id}/data/{source}/{record_id}/lineage")
    def get_lineage(
        cafe_id: str,
        source: str,
        record_id: str,
        user: dict[str, Any] = Depends(require_roles("owner", "manager")),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        accessible_cafe(cafe_id, user, repository)
        lineage = repository.lineage(cafe_id, source, record_id)
        if lineage is None:
            _problem(404, "lineage_not_found", "Lineage was not found for this record.")
        return _envelope(**lineage)

    @app.get("/api/runs")
    def list_runs(
        cafe_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        if cafe_id:
            accessible_cafe(cafe_id, user, repository)
        runs = repository.list_runs(cafe_id=cafe_id, limit=limit)
        if user["role"] != "owner":
            runs = [run for run in runs if run["cafe_id"] in user["cafe_ids"]]
        return _envelope(items=[_public_run(run) for run in runs])

    @app.post("/api/runs", status_code=202)
    def create_run(
        payload: RunCreateRequest,
        request: Request,
        user: dict[str, Any] = Depends(require_roles("owner", "manager")),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        cafe = accessible_cafe(payload.cafe_id, user, repository)
        queued = request.app.state.run_service.start(cafe, payload.target_week, user["id"])
        run = repository.get_run(queued["id"])
        return _envelope(run=_public_run(run or queued))

    @app.get("/api/runs/{run_id}")
    def get_run(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        return _envelope(run=_public_run(accessible_run(run_id, user, repository)))

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> StreamingResponse:
        accessible_run(run_id, user, repository)
        last_event_id = request.headers.get("Last-Event-ID")

        async def stream() -> AsyncIterator[str]:
            sent: set[str] = set()
            skip = bool(last_event_id)
            while True:
                events = repository.checkpoint_events(run_id)
                run = repository.get_run(run_id)
                if run and not events:
                    events = [
                        {
                            "schema_version": SCHEMA_VERSION,
                            "event_id": f"{run_id}:{run.get('updated_at')}",
                            "run_id": run_id,
                            "type": "run.status",
                            "at": run.get("updated_at"),
                            "stage": run.get("stage"),
                            "status": run.get("status"),
                            "message": f"Run is {str(run.get('status')).replace('_', ' ')}.",
                        }
                    ]
                for event in events:
                    event_id = event["event_id"]
                    if skip:
                        if event_id == last_event_id:
                            skip = False
                        continue
                    if event_id in sent:
                        continue
                    sent.add(event_id)
                    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    yield f"id: {event_id}\nevent: {event['type']}\ndata: {data}\n\n"
                if run is None or run["status"] in {
                    "succeeded", "failed", "rejected", "aborted", "partial", "waiting_review"
                }:
                    end = _envelope(run_id=run_id, type="end", status=run.get("status") if run else "unknown")
                    yield f"event: end\ndata: {json.dumps(end, separators=(',', ':'))}\n\n"
                    return
                if await request.is_disconnected():
                    return
                yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/findings")
    def get_findings(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        accessible_run(run_id, user, repository)
        return _envelope(run_id=run_id, items=repository.findings(run_id))

    @app.get("/api/runs/{run_id}/report")
    def get_report(
        run_id: str,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        accessible_run(run_id, user, repository)
        report = repository.report(run_id)
        if report is None:
            _problem(404, "report_not_found", "This run does not have a report yet.")
        return _envelope(**report)

    @app.get("/api/runs/{run_id}/report/location")
    def get_report_location(
        run_id: str,
        user: dict[str, Any] = Depends(require_roles("owner", "manager")),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        """Where this run's report files sit on the API host.

        The only endpoint that returns a real filesystem path, so it is
        role-gated rather than open to every signed-in account. `can_reveal`
        tells the UI whether POST .../reveal will do anything: opening a file
        manager is only meaningful when the browser and the API are the same
        machine, which we only assume in development.
        """
        accessible_run(run_id, user, repository)
        location = repository.report_location(run_id)
        if location is None:
            return _envelope(
                run_id=run_id, available=False, can_reveal=False, directory=None, files=[]
            )
        return _envelope(
            available=True,
            can_reveal=configured.environment == "development",
            **location,
        )

    @app.post("/api/runs/{run_id}/report/reveal")
    def reveal_report_location(
        run_id: str,
        user: dict[str, Any] = Depends(require_roles("owner", "manager")),
        repository: ArtifactRepository = Depends(artifacts),
    ) -> dict[str, Any]:
        """Open the run's report folder in the host's file manager.

        Development-only, and the path is never taken from the request: it is
        resolved from the run's own checkpoint and re-checked to sit inside
        outputs/reports. The launcher is called with an argument vector (no
        shell), so nothing here is interpolated into a command line.
        """
        accessible_run(run_id, user, repository)
        if configured.environment != "development":
            _problem(403, "reveal_not_available", "Revealing local files is a development-only action.")
        location = repository.report_location(run_id)
        if location is None:
            _problem(404, "report_files_not_found", "This run has no saved report files.")
        directory = location["directory"]
        if sys.platform == "win32":
            command = ["explorer.exe", directory]
        elif sys.platform == "darwin":
            command = ["open", directory]
        else:
            command = ["xdg-open", directory]
        try:
            # explorer.exe returns a non-zero exit code even when it succeeds,
            # so the return code is not a usable success signal here.
            subprocess.Popen(command, close_fds=True)
        except OSError as exc:
            _problem(500, "reveal_failed", f"Could not open the file manager: {exc}")
        return _envelope(revealed=True, directory=directory)

    @app.post("/api/runs/{run_id}/manager-review", status_code=201)
    def manager_review(
        run_id: str,
        payload: ManagerReviewRequest,
        # Managers by default. An owner is admitted only when the deployment
        # explicitly opts out of two-person review via
        # WADDEHHA_ALLOW_OWNER_SELF_REVIEW -- see the role check below, which
        # keeps the default 403 that segregation of duties depends on.
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
        db: ApiDatabase = Depends(database),
    ) -> dict[str, Any]:
        if user["role"] != "manager" and not (
            user["role"] == "owner" and configured.allow_owner_self_review
        ):
            _problem(403, "permission_denied", "Your role cannot perform this action.")
        run = accessible_run(run_id, user, repository)
        if repository.report(run_id) is None:
            _problem(409, "report_not_ready", "The report is not ready for review.")
        if run["status"] != "waiting_review" or run["report_state"] != "manager_review":
            _problem(409, "invalid_review_transition", "This run is not awaiting manager review.")
        try:
            review = db.add_review(run_id, user["id"], payload.decision, payload.comment)
        except ValueError:
            _problem(409, "manager_review_already_submitted", "Manager review was already submitted.")
        state = "owner_review" if payload.decision == "submit" else "manager_review"
        return _envelope(review=review, report_state=state)

    @app.post("/api/runs/{run_id}/decision", status_code=201)
    def owner_decision(
        run_id: str,
        payload: OwnerDecisionRequest,
        request: Request,
        user: dict[str, Any] = Depends(require_roles("owner")),
        repository: ArtifactRepository = Depends(artifacts),
        db: ApiDatabase = Depends(database),
    ) -> dict[str, Any]:
        run = accessible_run(run_id, user, repository)
        if run["status"] != "waiting_review" or run["report_state"] != "owner_review":
            _problem(409, "invalid_decision_transition", "This run is not awaiting an owner decision.")
        if not request.app.state.run_service.can_resume(run_id):
            _problem(409, "checkpoint_not_available", "This saved run cannot be resumed.")
        try:
            decision = db.add_decision(run_id, user["id"], payload.decision, payload.comment)
        except ValueError as exc:
            code = str(exc)
            message = (
                "Submit the manager review before the owner decision."
                if code == "manager_review_required"
                else "An owner decision was already recorded for this run."
            )
            _problem(409, code, message)
        request.app.state.run_service.resume(run_id, payload.decision)
        state = {"approve": "approved", "reject": "rejected", "edit": "analyzing"}[payload.decision]
        return _envelope(decision=decision, report_state=state)

    @app.post("/api/conversations", status_code=201)
    def create_conversation(
        payload: ConversationCreateRequest,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
        db: ApiDatabase = Depends(database),
    ) -> dict[str, Any]:
        accessible_cafe(payload.cafe_id, user, repository)
        if payload.run_id:
            run = accessible_run(payload.run_id, user, repository)
            if run["cafe_id"] != payload.cafe_id:
                _problem(409, "run_cafe_mismatch", "The run does not belong to this café.")
        conversation = db.create_conversation(payload.cafe_id, payload.run_id, user["id"])
        return _envelope(conversation=conversation)

    @app.get("/api/conversations")
    def list_conversations(
        user: dict[str, Any] = Depends(current_user), db: ApiDatabase = Depends(database)
    ) -> dict[str, Any]:
        return _envelope(items=db.list_conversations(user["id"]))

    @app.post("/api/conversations/{conversation_id}/messages", status_code=201)
    def create_message(
        conversation_id: str,
        payload: MessageCreateRequest,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
        repository: ArtifactRepository = Depends(artifacts),
        db: ApiDatabase = Depends(database),
    ) -> dict[str, Any]:
        conversation = db.get_conversation(conversation_id, user["id"])
        if conversation is None:
            _problem(404, "conversation_not_found", "Conversation not found.")
        accessible_cafe(conversation["cafe_id"], user, repository)
        user_message = db.add_message(conversation_id, "user", payload.message)
        messages = db.list_messages(conversation_id)
        
        # Inject AI settings into os.environ for the chat agent
        request.app.state.ai_settings.inject_environment()
        
        answer, citations, grounded_run_id = grounded_answer(
            repository, conversation["cafe_id"], conversation["run_id"], messages
        )
        assistant_message = db.add_message(conversation_id, "assistant", answer, citations)
        return _envelope(
            conversation_id=conversation_id,
            run_id=grounded_run_id,
            user_message=user_message,
            assistant_message=assistant_message,
            message=answer,
            citations=citations
        )

    return app


app = create_app()
