import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 45_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:13000",
    channel: "msedge",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "edge-desktop",
      use: { ...devices["Desktop Edge"] },
    },
  ],
});
