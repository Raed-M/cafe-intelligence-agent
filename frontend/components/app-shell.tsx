"use client";

import {
  Activity,
  BarChart3,
  ChevronLeft,
  Command,
  Database,
  FileCheck2,
  Languages,
  BrainCircuit,
  LogOut,
  Menu,
  MessageCircle,
  PanelRightClose,
  Search,
  Send,
  ShieldCheck,
  UserRoundCheck,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { FormEvent, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatAnswer } from "@/lib/types";
import { displayError } from "@/lib/format";
import { useWorkspace } from "@/components/providers";
import { StatePanel } from "@/components/ui";

type NavKey = "weekly" | "data" | "agents" | "reports" | "requests" | "ai";
type DashboardSection = "weekly" | "agents" | "reports";

const nav = [
  { key: "weekly", sectionId: "weekly-story", href: "/dashboard", ar: "القصة الأسبوعية", en: "Weekly story", icon: BarChart3, groupAr: "نظرة عامة", groupEn: "Overview" },
  { key: "data", href: "/data", ar: "مستكشف البيانات", en: "Data explorer", icon: Database, groupAr: "البيانات", groupEn: "Data" },
  { key: "agents", sectionId: "agents", href: "/dashboard#agents", ar: "الوكلاء", en: "Agents", icon: Activity, groupAr: "التحليل", groupEn: "Analysis" },
  { key: "reports", sectionId: "reports", href: "/dashboard#reports", ar: "التقارير والموافقات", en: "Reports & approvals", icon: FileCheck2, groupAr: "التحكم", groupEn: "Control" },
  { key: "ai", href: "/ai-connections", ar: "اتصالات الذكاء الاصطناعي", en: "AI connections", icon: BrainCircuit, groupAr: "التحكم", groupEn: "Control" },
  { key: "requests", href: "/access-requests", ar: "طلبات الوصول", en: "Access requests", icon: UserRoundCheck, groupAr: "التحكم", groupEn: "Control" },
] satisfies Array<{ key: NavKey; sectionId?: string; href: string; ar: string; en: string; icon: typeof BarChart3; groupAr: string; groupEn: string }>;

const roleName = (role: string, locale: "ar" | "en") => {
  const names: Record<string, [string, string]> = { owner: ["المالك", "Owner"], manager: ["المدير", "Manager"], employee: ["الموظف", "Employee"], demo_visitor: ["زائر العرض", "Demo visitor"] };
  return names[role]?.[locale === "ar" ? 0 : 1] ?? role;
};

function BrandMark({ compact }: { compact: boolean }) {
  return <Link className="brand" href="/dashboard" aria-label="وضحها — WADDEHHA"><span className="brand-mark" aria-hidden="true"><span /></span>{!compact && <span><b>وضّحها</b><small>WADDEHHA · CAFÉ INTELLIGENCE</small></span>}</Link>;
}

function ChatMessage({ role, content, citations }: { role: "user" | "assistant"; content: string; citations?: Array<{ url?: string; label?: string }> }) {
  const Markdown = require("react-markdown").default;
  const remarkGfm = require("remark-gfm").default;
  if (role === "user") return <div className="chat-msg chat-msg-user"><p>{content}</p></div>;
  return <div className="chat-msg chat-msg-assistant">
    <Markdown remarkPlugins={[remarkGfm]} components={{
      a: ({ href, children, ...props }: any) => <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>,
      table: ({ children, ...props }: any) => <div className="chat-table-wrap"><table {...props}>{children}</table></div>,
    }}>{content}</Markdown>
    {citations && citations.length > 0 && <div className="chat-citations">{citations.map((c, i) => <a key={i} href={c.url} target="_blank" rel="noopener noreferrer" className="chat-cite-link">{c.label || c.url}</a>)}</div>}
  </div>;
}

type ChatMsg = { role: "user" | "assistant"; content: string; citations?: Array<{ url?: string; label?: string }> };

function ChatComposer() {
  const { locale, cafeId } = useWorkspace();
  const [open, setOpen] = useState(false);
  const [suppressed, setSuppressed] = useState(false);
  const [message, setMessage] = useState("");
  const [conversationId, setConversationId] = useState("");
  const [thread, setThread] = useState<ChatMsg[]>([]);
  const [error, setError] = useState("");
  const shellRef = useRef<HTMLElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const mutation = useMutation({
    mutationFn: async (text: string) => {
      const conversation = conversationId ? { id: conversationId } : await api.createConversation(cafeId);
      if (!conversationId) setConversationId(conversation.id);
      return api.sendMessage(conversation.id, text);
    },
    onSuccess(data) {
      const reply = data.answer || data.message || "";
      const cites: Array<{ url?: string; label?: string }> = (data.citations || []).map(c =>
        typeof c === "string" ? { url: c, label: c } : { url: (c as any).url || (c as any).id, label: (c as any).label || (c as any).id }
      );
      setThread(prev => [...prev, { role: "assistant", content: reply, citations: cites }]);
      setMessage(""); setError(""); setOpen(true);
    },
    onError(reason) { setError(displayError(reason, locale === "ar" ? "تعذر إرسال السؤال." : "Could not send question.")); setOpen(true); },
  });

  useEffect(() => {
    const closeOutside = (event: PointerEvent) => {
      if (shellRef.current?.contains(event.target as Node)) return;
      setOpen(false);
      inputRef.current?.blur();
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      inputRef.current?.blur();
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    const timer = window.setTimeout(() => inputRef.current?.focus({ preventScroll: true }), 50);
    return () => window.clearTimeout(timer);
  }, [open]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [thread, mutation.isPending]);

  const reveal = () => {
    setSuppressed(false);
    setOpen(true);
  };
  const close = () => {
    setSuppressed(true);
    setOpen(false);
    inputRef.current?.blur();
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const text = message.trim();
    if (!text || !cafeId || mutation.isPending) return;
    setThread(prev => [...prev, { role: "user", content: text }]);
    mutation.mutate(text);
  };

  const hasResult = thread.length > 0 || mutation.isPending || Boolean(error);
  return <section ref={shellRef} className={`chat-shell ${open ? "chat-open" : ""} ${suppressed ? "chat-suppressed" : ""}`} onMouseEnter={() => { if (!open) setSuppressed(false); }} aria-label={locale === "ar" ? "اسأل بياناتك" : "Ask your data"}>
    <button className="chat-peek" type="button" onPointerDown={(event) => { event.preventDefault(); reveal(); }} onClick={reveal} aria-expanded={open} aria-controls="chat-composer-surface">
      <MessageCircle aria-hidden="true" /><span>{locale === "ar" ? "اسأل بياناتك" : "Ask your data"}</span>
    </button>
    <div className="chat-surface" id="chat-composer-surface">
      {hasResult && <div className="chat-result" ref={scrollRef} aria-live="polite">
        <div className="chat-result-head"><span><ShieldCheck aria-hidden="true" />{locale === "ar" ? "المساعد التشغيلي" : "Operational Assistant"}</span><button type="button" className="icon-button" onClick={close} aria-label={locale === "ar" ? "تصغير" : "Collapse"}><X /></button></div>
        <div className="chat-thread">
          {thread.map((msg, i) => <ChatMessage key={i} role={msg.role} content={msg.content} citations={msg.citations} />)}
          {mutation.isPending && <div className="chat-msg chat-msg-assistant"><StatePanel kind="loading" title={locale === "ar" ? "يحلل البيانات ويجهز الرد…" : "Analyzing data and preparing response…"} /></div>}
          {error && <StatePanel kind="error" title={locale === "ar" ? "لم تصل إجابة" : "No answer received"} body={error} />}
        </div>
      </div>}
      <form className="chat-form" onSubmit={submit} onFocus={() => setOpen(true)}>
        <MessageCircle aria-hidden="true" />
        <textarea ref={inputRef} value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(event as unknown as FormEvent); } }} rows={1} placeholder={locale === "ar" ? "اسأل عن المبيعات، المخزون، التقييمات، أو الترندات…" : "Ask about sales, inventory, reviews, or trends…"} aria-label={locale === "ar" ? "اكتب سؤالك" : "Type your question"} disabled={!cafeId || mutation.isPending} />
        <span className="context-chip">{locale === "ar" ? "سياق المقهى الحالي" : "Current café context"}</span>
        <button className="send-button" type="submit" disabled={!message.trim() || !cafeId || mutation.isPending} aria-label={locale === "ar" ? "إرسال" : "Send"}><Send /></button>
      </form>
    </div>
  </section>;
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { locale, setLocale, viewMode, setViewMode, user, cafes, cafe, cafeId, setCafeId, loading, authError } = useWorkspace();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [commandSearch, setCommandSearch] = useState("");
  const [activeDashboardSection, setActiveDashboardSection] = useState<DashboardSection>("weekly");

  useEffect(() => { if (authError) router.replace("/login"); }, [authError, router]);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); setCommandOpen(true); }
      if (event.key === "Escape") { setCommandOpen(false); setMobileOpen(false); }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  useEffect(() => {
    if (pathname !== "/dashboard") return;
    let frame = 0;
    const updateActiveSection = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        const activationLine = Math.min(window.innerHeight * 0.34, 250);
        let next: DashboardSection = "weekly";
        for (const [key, id] of [["weekly", "weekly-story"], ["agents", "agents"], ["reports", "reports"]] as const) {
          const section = document.getElementById(id);
          if (section && section.getBoundingClientRect().top <= activationLine) next = key;
        }
        if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) next = "reports";
        setActiveDashboardSection(next);
      });
    };
    updateActiveSection();
    window.addEventListener("scroll", updateActiveSection, { passive: true });
    window.addEventListener("hashchange", updateActiveSection);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", updateActiveSection);
      window.removeEventListener("hashchange", updateActiveSection);
    };
  }, [pathname]);

  useEffect(() => {
    if (pathname !== "/dashboard") window.scrollTo({ top: 0, behavior: "instant" });
  }, [pathname]);

  const visibleNav = useMemo(() => nav.filter((item) => !["requests", "ai"].includes(item.key) || user?.role === "owner"), [user?.role]);
  const filteredNav = useMemo(() => visibleNav.filter((item) => `${item.ar} ${item.en}`.toLowerCase().includes(commandSearch.toLowerCase())), [commandSearch, visibleNav]);
  if (loading) return <main className="auth-state"><div className="large-pearl" /><StatePanel kind="loading" title={locale === "ar" ? "يتم فتح مساحة العمل" : "Opening workspace"} body={locale === "ar" ? "نتحقق من الجلسة والمقاهي المصرّح بها." : "Checking your session and authorized cafés."} /></main>;
  if (authError) return null;

  const logout = async () => { await api.logout().catch(() => undefined); queryClient.clear(); router.replace("/login"); };
  const grouped = visibleNav.reduce<Record<string, typeof nav>>((result, item) => { const key = locale === "ar" ? item.groupAr : item.groupEn; (result[key] ||= []).push(item); return result; }, {});
  const navIsActive = (key: NavKey) => {
    if (pathname === "/dashboard") return (key === "weekly" || key === "agents" || key === "reports") && activeDashboardSection === key;
    if (key === "data") return pathname.startsWith("/data");
    if (key === "agents") return pathname.startsWith("/runs") || pathname.startsWith("/findings");
    if (key === "reports") return pathname.startsWith("/reports");
    if (key === "ai") return pathname.startsWith("/ai-connections");
    if (key === "requests") return pathname.startsWith("/access-requests");
    return false;
  };
  const navigateToDashboardSection = (event: ReactMouseEvent<HTMLAnchorElement>, key: DashboardSection, sectionId?: string) => {
    setMobileOpen(false);
    if (!sectionId) return;
    setActiveDashboardSection(key);
    if (pathname !== "/dashboard") return;
    const section = document.getElementById(sectionId);
    if (!section) return;
    event.preventDefault();
    window.history.replaceState(null, "", `/dashboard${key === "weekly" ? "" : `#${sectionId}`}`);
    section.scrollIntoView({ block: "start", behavior: "instant" as ScrollBehavior });
  };

  return <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
    <a className="skip-link" href="#main-content">{locale === "ar" ? "انتقل إلى المحتوى" : "Skip to content"}</a>
    <aside className={`sidebar ${mobileOpen ? "sidebar-mobile-open" : ""}`}>
      <div className="sidebar-brand"><BrandMark compact={collapsed} /><button type="button" className="icon-button sidebar-close" onClick={() => setMobileOpen(false)} aria-label={locale === "ar" ? "إغلاق القائمة" : "Close menu"}><X /></button></div>
      <nav aria-label={locale === "ar" ? "التنقل الرئيسي" : "Primary navigation"}>
        {Object.entries(grouped).map(([group, links]) => <div className="nav-group" key={group}>{!collapsed && <span className="nav-heading">{group}</span>}{links.map(({ key, sectionId, href, ar, en, icon: Icon }) => {
          const active = navIsActive(key);
          return <Link className={`nav-link ${active ? "active" : ""}`} href={href} key={href} aria-current={active ? "page" : undefined} title={collapsed ? (locale === "ar" ? ar : en) : undefined} onClick={(event) => sectionId ? navigateToDashboardSection(event, key as DashboardSection, sectionId) : setMobileOpen(false)}><Icon aria-hidden="true" /><span>{locale === "ar" ? ar : en}</span></Link>;
        })}</div>)}
      </nav>
      <div className="sidebar-foot">
        {!collapsed && <div className="identity"><span>{user?.display_name || user?.email}</span><small>{roleName(user?.role || "", locale)} · <span className="honest-label">{locale === "ar" ? "جلسة محلية" : "Local session"}</span></small></div>}
        <button className="nav-link danger" type="button" onClick={logout} title={locale === "ar" ? "تسجيل الخروج" : "Sign out"}><LogOut /><span>{locale === "ar" ? "تسجيل الخروج" : "Sign out"}</span></button>
      </div>
    </aside>

    <div className="workspace">
      <header className="topbar">
        <button className="icon-button mobile-menu" type="button" onClick={() => setMobileOpen(true)} aria-label={locale === "ar" ? "فتح القائمة" : "Open menu"}><Menu /></button>
        <button className="icon-button collapse-button" type="button" onClick={() => setCollapsed((value) => !value)} aria-label={locale === "ar" ? "طي القائمة الجانبية" : "Collapse sidebar"}><PanelRightClose /></button>
        <div className="cafe-select"><label htmlFor="cafe-select">{locale === "ar" ? "المقهى / الفرع" : "Café / branch"}</label><select id="cafe-select" value={cafeId} onChange={(event) => setCafeId(event.target.value)} disabled={!cafes.length}>{cafes.length ? cafes.map((option) => <option value={option.id} key={option.id}>{option.name}{option.city ? ` · ${option.city}` : ""}</option>) : <option value="">{locale === "ar" ? "لا توجد مقاهٍ مصرّح بها" : "No authorized cafés"}</option>}</select></div>
        <div className="topbar-actions">
          <div className="segmented" aria-label={locale === "ar" ? "مستوى التفاصيل" : "Detail level"}><button type="button" className={viewMode === "executive" ? "selected" : ""} onClick={() => setViewMode("executive")}>{locale === "ar" ? "تنفيذي" : "Executive"}</button><button type="button" className={viewMode === "technical" ? "selected" : ""} onClick={() => setViewMode("technical")}>{locale === "ar" ? "تقني" : "Technical"}</button></div>
          <button className="tool-button command-trigger" type="button" onClick={() => setCommandOpen(true)}><Command /> <span>{locale === "ar" ? "الأوامر" : "Commands"}</span><kbd>⌘K</kbd></button>
          <button className="tool-button" type="button" onClick={() => setLocale(locale === "ar" ? "en" : "ar")}><Languages />{locale === "ar" ? "EN" : "عربي"}</button>
        </div>
      </header>
      <main id="main-content" tabIndex={-1} className="main-content">{!cafe && !loading ? <StatePanel kind="empty" title={locale === "ar" ? "لا توجد مساحة عمل متاحة" : "No workspace available"} body={locale === "ar" ? "اطلب من المالك منح حسابك صلاحية على مقهى." : "Ask an owner to grant your account access to a café."} /> : children}</main>
      <ChatComposer />
    </div>

    {mobileOpen && <button className="scrim" type="button" onClick={() => setMobileOpen(false)} aria-label={locale === "ar" ? "إغلاق القائمة" : "Close menu"} />}
    {commandOpen && <div className="modal-layer" role="dialog" aria-modal="true" aria-labelledby="command-title"><button className="modal-scrim" onClick={() => setCommandOpen(false)} aria-label={locale === "ar" ? "إغلاق" : "Close"} /><div className="command-panel"><div className="command-search"><Search /><label className="sr-only" htmlFor="command-search">{locale === "ar" ? "البحث في الأوامر" : "Search commands"}</label><input autoFocus id="command-search" value={commandSearch} onChange={(event) => setCommandSearch(event.target.value)} placeholder={locale === "ar" ? "ابحث عن صفحة…" : "Find a page…"} /><button className="icon-button" type="button" onClick={() => setCommandOpen(false)}><X /></button></div><h2 id="command-title">{locale === "ar" ? "انتقل إلى" : "Go to"}</h2>{filteredNav.map(({ href, ar, en, icon: Icon }) => <button type="button" className="command-item" key={href} onClick={() => { router.push(href); setCommandOpen(false); }}><Icon /><span>{locale === "ar" ? ar : en}</span><ChevronLeft /></button>)}</div></div>}
  </div>;
}
