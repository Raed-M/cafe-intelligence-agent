"use client";

import { FormEvent, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowRight, BriefcaseBusiness, Eye, EyeOff, Languages, UserRound, UsersRound } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { displayError } from "@/lib/format";
import { useWorkspace } from "@/components/providers";
import { StatePanel } from "@/components/ui";

export function SignupView() {
  const { locale, setLocale } = useWorkspace();
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState<"manager" | "employee">("employee");
  const [showPassword, setShowPassword] = useState(false);
  const signup = useMutation({ mutationFn: () => api.signup({ display_name: displayName.trim(), email: email.trim(), password, requested_role: role }) });
  const passwordsMatch = password === confirmPassword;
  const ready = displayName.trim().length >= 2 && email.includes("@") && password.length >= 8 && passwordsMatch;
  const submit = (event: FormEvent) => { event.preventDefault(); if (ready && !signup.isPending) signup.mutate(); };

  return <main className="login-page signup-page">
    <button className="language-float" type="button" onClick={() => setLocale(locale === "ar" ? "en" : "ar")}><Languages />{locale === "ar" ? "English" : "العربية"}</button>
    <section className="login-atmosphere" aria-labelledby="signup-story-title"><div className="coast-lines" aria-hidden="true"><span /><span /><span /></div><div className="login-pearl" aria-hidden="true"><span /></div><div className="login-story"><span className="eyebrow">CONTROLLED ACCESS</span><h1 id="signup-story-title">Request the right access.</h1><p>Choose the role you need. The owner sees your request and must approve it before you can sign in.</p><div className="trust-row"><span><UserRound />Real identity</span><span><UsersRound />Owner approval</span></div></div></section>
    <section className="login-panel"><div className="login-form-wrap"><div className="login-brand"><span className="brand-mark" aria-hidden="true"><span /></span><div><b>وضّحها</b><small>WADDEHHA</small></div></div>{signup.isSuccess ? <div className="signup-success"><StatePanel kind="empty" title="Request sent to the owner" body={`Your ${role} account is pending approval. You can sign in after Admin accepts it.`} /><Link className="primary-button login-button" href="/login">Return to sign in<ArrowRight /></Link></div> : <><h2>Request access</h2><p>Enter your information, then choose Manager or Employee.</p><form onSubmit={submit} noValidate><div className="field"><label htmlFor="signup-name">Full name</label><input id="signup-name" autoComplete="name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} required /></div><div className="field"><label htmlFor="signup-email">Email address</label><input id="signup-email" type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></div><fieldset className="role-choice"><legend>Requested role</legend><button type="button" className={role === "manager" ? "selected" : ""} aria-pressed={role === "manager"} onClick={() => setRole("manager")}><BriefcaseBusiness /><span><strong>Manager</strong><small>Run analysis, inspect data, and submit reports.</small></span></button><button type="button" className={role === "employee" ? "selected" : ""} aria-pressed={role === "employee"} onClick={() => setRole("employee")}><UserRound /><span><strong>Employee</strong><small>Read assigned café stories, findings, and reports.</small></span></button></fieldset><div className="field"><label htmlFor="signup-password">Password</label><div className="password-field"><input id="signup-password" type={showPassword ? "text" : "password"} autoComplete="new-password" value={password} onChange={(event) => setPassword(event.target.value)} aria-describedby="signup-password-help" required /><button className="icon-button" type="button" onClick={() => setShowPassword((value) => !value)} aria-label={showPassword ? "Hide password" : "Show password"}>{showPassword ? <EyeOff /> : <Eye />}</button></div><small id="signup-password-help">Use at least 8 characters.</small></div><div className="field"><label htmlFor="signup-confirm">Confirm password</label><input id="signup-confirm" type={showPassword ? "text" : "password"} autoComplete="new-password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} aria-invalid={Boolean(confirmPassword && !passwordsMatch)} required />{confirmPassword && !passwordsMatch && <small className="field-error">Passwords do not match.</small>}</div>{signup.isError && <div className="form-error" role="alert"><strong>Could not send request</strong><span>{displayError(signup.error, "Check your details and try again.")}</span></div>}<button className="primary-button login-button" type="submit" disabled={!ready || signup.isPending}>{signup.isPending ? "Sending request…" : "Send access request"}<ArrowRight /></button></form><p className="auth-switch">Already approved? <Link href="/login">Sign in</Link></p></>}</div></section>
  </main>;
}
