import { FindingView } from "@/components/finding-view";
export default async function FindingPage({ params }: { params: Promise<{ runId: string; findingId: string }> }) { const { runId, findingId } = await params; return <FindingView runId={runId} findingId={findingId} />; }
