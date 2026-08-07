import type { Metadata } from "next";
import { DashboardView } from "@/components/dashboard-view";
export const metadata: Metadata = { title: "القصة الأسبوعية" };
export default function DashboardPage() { return <DashboardView />; }
