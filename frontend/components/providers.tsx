"use client";

import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import type { Cafe, User } from "@/lib/types";

type Locale = "ar" | "en";
type ViewMode = "executive" | "technical";
interface WorkspaceContextValue {
  locale: Locale;
  setLocale: (value: Locale) => void;
  viewMode: ViewMode;
  setViewMode: (value: ViewMode) => void;
  user?: User;
  cafes: Cafe[];
  cafe?: Cafe;
  cafeId: string;
  setCafeId: (value: string) => void;
  loading: boolean;
  authError: boolean;
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");
  const [viewMode, setViewMode] = useState<ViewMode>("executive");
  const [cafeId, setCafeIdState] = useState("");
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  const cafesQuery = useQuery({ queryKey: ["cafes"], queryFn: api.cafes, enabled: Boolean(me.data), retry: 1 });

  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dir = locale === "ar" ? "rtl" : "ltr";
  }, [locale]);

  const effectiveCafeId = cafeId || cafesQuery.data?.[0]?.id || "";

  const value = useMemo<WorkspaceContextValue>(() => ({
    locale,
    setLocale(value) { setLocaleState(value); window.localStorage.setItem("waddehha.locale", value); },
    viewMode,
    setViewMode,
    user: me.data,
    cafes: cafesQuery.data ?? [],
    cafe: cafesQuery.data?.find((candidate) => candidate.id === effectiveCafeId),
    cafeId: effectiveCafeId,
    setCafeId(value) { setCafeIdState(value); window.localStorage.setItem("waddehha.cafe", value); },
    loading: me.isLoading || cafesQuery.isLoading,
    authError: me.isError,
  }), [locale, viewMode, me.data, me.isLoading, me.isError, cafesQuery.data, cafesQuery.isLoading, effectiveCafeId]);

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { staleTime: 15_000, refetchOnWindowFocus: false } } }));
  return <QueryClientProvider client={queryClient}><WorkspaceProvider>{children}</WorkspaceProvider></QueryClientProvider>;
}

export function useWorkspace() {
  const context = useContext(WorkspaceContext);
  if (!context) throw new Error("useWorkspace must be used within Providers");
  return context;
}
