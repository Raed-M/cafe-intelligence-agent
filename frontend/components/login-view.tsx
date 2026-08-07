"use client";

import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Eye, EyeOff, Languages, LockKeyhole, Radio, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { displayError } from "@/lib/format";
import { useWorkspace } from "@/components/providers";

export function LoginView() {
  const { locale, setLocale } = useWorkspace();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const health = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const response = await fetch("/api/health");
      if (!response.ok) throw new Error("offline");
      return response.json() as Promise<{ status?: string; version?: string }>;
    },
    retry: 1,
  });
  const login = useMutation({
    mutationFn: () => api.login(email.trim(), password),
    onSuccess(data) { queryClient.setQueryData(["me"], data.user); router.replace("/dashboard"); },
  });
  const Arrow = locale === "ar" ? ArrowLeft : ArrowRight;
  const submit = (event: FormEvent) => { event.preventDefault(); if (email.trim() && password && !login.isPending) login.mutate(); };

  return <main className="login-page">
    <button className="language-float" type="button" onClick={() => setLocale(locale === "ar" ? "en" : "ar")}><Languages />{locale === "ar" ? "English" : "العربية"}</button>
    <section className="login-atmosphere" aria-labelledby="login-story-title">
      <div className="coast-lines" aria-hidden="true"><span /><span /><span /></div>
      <div className="login-pearl" aria-hidden="true"><span /></div>
      <div className="login-story"><span className="eyebrow">WADDEHHA · CAFÉ INTELLIGENCE</span><h1 id="login-story-title">{locale === "ar" ? "من ضجيج البيانات،\nتظهر الحقيقة." : "From data noise,\nclarity surfaces."}</h1><p>{locale === "ar" ? "مساحة تشغيل تربط كل قرار بحسابه ومصدره ومراجعة بشرية." : "An operations space that ties every decision to its calculation, source, and human review."}</p><div className="trust-row"><span><ShieldCheck />{locale === "ar" ? "أدلة قبل الادعاء" : "Evidence before claims"}</span><span><LockKeyhole />{locale === "ar" ? "جلسة HttpOnly" : "HttpOnly session"}</span></div></div>
    </section>
    <section className="login-panel"><div className="login-form-wrap">
      <div className="login-brand"><span className="brand-mark" aria-hidden="true"><span /></span><div><b>وضّحها</b><small>WADDEHHA</small></div></div>
      <div className="api-status"><span className={`stream-dot ${health.isSuccess ? "stream-open" : health.isLoading ? "stream-connecting" : "stream-error"}`} /><span>{health.isSuccess ? (locale === "ar" ? "واجهة API متصلة" : "API connected") : health.isLoading ? (locale === "ar" ? "فحص API…" : "Checking API…") : (locale === "ar" ? "واجهة API غير متاحة" : "API unavailable")}</span>{health.data?.version && <code>v{health.data.version}</code>}</div>
      <h2>{locale === "ar" ? "افتح مساحة العمل" : "Open your workspace"}</h2>
      <p>{locale === "ar" ? "دخول المالك المحلي: admin وكلمة المرور admin. الحسابات الأخرى تحتاج موافقة المالك." : "Local owner login: admin with password admin. Other accounts require owner approval."}</p>
      <form onSubmit={submit} noValidate>
        <div className="field"><label htmlFor="email">{locale === "ar" ? "البريد أو اسم المستخدم" : "Email or username"}</label><input id="email" name="email" type="text" autoComplete="username" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="admin" required /></div>
        <div className="field"><label htmlFor="password">{locale === "ar" ? "كلمة المرور" : "Password"}</label><div className="password-field"><input id="password" name="password" type={showPassword ? "text" : "password"} autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="admin" required /><button type="button" className="icon-button" onClick={() => setShowPassword((value) => !value)} aria-label={showPassword ? (locale === "ar" ? "إخفاء كلمة المرور" : "Hide password") : (locale === "ar" ? "إظهار كلمة المرور" : "Show password")}>{showPassword ? <EyeOff /> : <Eye />}</button></div></div>
        {login.isError && <div className="form-error" role="alert"><strong>{locale === "ar" ? "لم ينجح تسجيل الدخول" : "Sign-in failed"}</strong><span>{displayError(login.error, locale === "ar" ? "تحقق من بيانات الحساب أو حالة موافقته." : "Check the account details or approval status.")}</span></div>}
        <button className="primary-button login-button" type="submit" disabled={!email.trim() || !password || login.isPending}>{login.isPending ? (locale === "ar" ? "جارٍ التحقق…" : "Checking…") : (locale === "ar" ? "تسجيل الدخول" : "Sign in")}<Arrow /></button>
      </form>
      <p className="auth-switch">{locale === "ar" ? "تحتاج حسابًا؟" : "Need an account?"} <Link href="/signup">{locale === "ar" ? "أرسل طلب وصول" : "Request access"}</Link></p>
      <div className="role-note"><Radio /><div><strong>{locale === "ar" ? "صلاحيات بموافقة المالك" : "Owner-approved roles"}</strong><span>{locale === "ar" ? "يختار المتقدم Manager أو Employee، ولا يستطيع الدخول حتى يقبل المالك الطلب." : "Applicants choose Manager or Employee and cannot sign in until the owner accepts the request."}</span></div></div>
    </div></section>
  </main>;
}
