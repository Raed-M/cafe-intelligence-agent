import type { Metadata } from "next";
import { AiConnectionsView } from "@/components/ai-connections-view";

export const metadata: Metadata = { title: "AI connections · Waddehha" };

export default function AiConnectionsPage() {
  return <AiConnectionsView />;
}
