import type { Metadata } from "next";
import { SignupView } from "@/components/signup-view";

export const metadata: Metadata = { title: "Request access · Waddehha" };
export default function SignupPage() { return <SignupView />; }
