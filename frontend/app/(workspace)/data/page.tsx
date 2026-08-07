import { Suspense } from "react";
import { DataExplorer } from "@/components/data-explorer";
import { Skeleton } from "@/components/ui";
export default function DataPage() { return <Suspense fallback={<Skeleton lines={8} />}><DataExplorer /></Suspense>; }
