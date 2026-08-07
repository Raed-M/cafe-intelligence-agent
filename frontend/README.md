# وضّحها frontend

Arabic-first Next.js App Router interface for the WADDEHHA FastAPI service.

## Run locally

```powershell
npm.cmd install
npm.cmd run dev
```

The frontend proxies `/api/*` to `API_ORIGIN` (default `http://127.0.0.1:8000`). Copy `.env.example` to `.env.local` only if the API runs elsewhere. API keys remain server-side and must never use a `NEXT_PUBLIC_` variable.

## Development login

The browser always uses the real `POST /api/auth/login` email/password flow. The backend creates no accounts by default. For deterministic local accounts, start the API with:

- `WADDEHHA_ENV=development`
- `WADDEHHA_DEV_SEED_USERS=1`
- strong values for `WADDEHHA_DEV_OWNER_PASSWORD`, `WADDEHHA_DEV_MANAGER_PASSWORD`, and `WADDEHHA_DEV_EMPLOYEE_PASSWORD`

Optional email variables default to `owner@waddehha.local`, `manager@waddehha.local`, and `employee@waddehha.local`. Passwords are intentionally never displayed or stored in the frontend.

Test evidence is excluded by default. The backend only exposes its deterministic test workspace when `WADDEHHA_INCLUDE_TEST_EVIDENCE=1` is explicitly set for local development.

## Checks

```powershell
npm.cmd run lint
npm.cmd run typecheck
npm.cmd run test
npm.cmd run build
npm.cmd audit --omit=dev
```
