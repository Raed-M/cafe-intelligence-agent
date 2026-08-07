import type { Metadata } from "next";
import { AccessRequestsView } from "@/components/access-requests-view";

export const metadata: Metadata = { title: "Access requests · Waddehha" };
export default function AccessRequestsPage() { return <AccessRequestsView />; }
