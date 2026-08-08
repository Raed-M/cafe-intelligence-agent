import { mkdir } from "node:fs/promises";
import path from "node:path";
import { expect, test } from "@playwright/test";

const repositoryRoot = path.resolve(process.cwd(), "..");
const evidenceDirectory = path.join(repositoryRoot, "outputs", "test_evidence", "ui");

test("admin controls access and can use the English workspace interactions", async ({ page }) => {
  await mkdir(evidenceDirectory, { recursive: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  const applicantEmail = `e2e.manager.${Date.now()}@example.com`;

  await page.goto("/signup");
  await expect(page.locator("html")).toHaveAttribute("dir", "ltr");
  await page.getByLabel("Full name").fill("E2E Manager");
  await page.getByLabel("Email address").fill(applicantEmail);
  await page.getByRole("button", { name: /Manager/ }).click();
  await page.getByLabel("Password", { exact: true }).fill("E2E-manager-pass-123");
  await page.getByLabel("Confirm password").fill("E2E-manager-pass-123");
  await page.getByRole("button", { name: "Send access request" }).click();
  await expect(page.getByText("Request sent to the owner")).toBeVisible();
  await page.getByRole("link", { name: /Return to sign in/ }).click();

  await expect(page.getByText("API connected", { exact: true })).toBeVisible();
  await page.getByLabel("Email or username").fill(applicantEmail);
  await page.getByLabel("Password", { exact: true }).fill("E2E-manager-pass-123");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByText("Your access request is waiting for owner approval.")).toBeVisible();

  await page.getByLabel("Email or username").fill("admin");
  await page.getByLabel("Password", { exact: true }).fill("admin");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page).toHaveURL(/\/dashboard$/);
  await expect(page.getByRole("heading", { name: "Weekly story", level: 1 })).toBeVisible();
  await expect(page.locator(".nav-link.active")).toHaveCount(1);
  await expect(page.locator(".nav-link.active")).toContainText("Weekly story");

  await page.getByRole("link", { name: "Access requests" }).click();
  const requestCard = page.locator(".request-card").filter({ hasText: applicantEmail });
  await expect(requestCard).toBeVisible();
  await requestCard.getByRole("button", { name: "Approve" }).click();
  await expect(requestCard).toBeHidden();

  await page.getByRole("link", { name: "Weekly story" }).click();
  await page.getByRole("button", { name: "Technical" }).click();
  await page.getByRole("link", { name: "Agents" }).click();
  await expect(page.locator(".nav-link.active")).toHaveCount(1);
  await expect(page.locator(".nav-link.active")).toContainText("Agents");
  await expect(page.getByRole("heading", { name: "Live execution map" })).toBeVisible();
  await expect(page.locator(".workflow-step[aria-current='step'], .workflow-step button[aria-current='step']")).toHaveCount(1);

  await page.getByRole("link", { name: "Reports & approvals" }).click();
  await expect(page.locator(".nav-link.active")).toHaveCount(1);
  await expect(page.locator(".nav-link.active")).toContainText("Reports & approvals");
  await page.locator("#agents").scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy({ top: -120, behavior: "instant" }));
  await expect(page.locator(".nav-link.active")).toContainText("Agents");

  const chatPeek = page.getByRole("button", { name: "Ask your data" });
  await chatPeek.click();
  const chatInput = page.getByLabel("Type your question");
  await expect(chatInput).toBeFocused();
  await chatInput.fill("What happened last week?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.locator(".chat-result .spin")).toBeHidden({ timeout: 15_000 });
  await page.getByRole("button", { name: "Collapse", exact: true }).click();
  await expect(page.locator(".chat-surface")).toBeHidden();

  await page.getByRole("link", { name: "AI connections" }).click();
  await expect(page.getByRole("heading", { name: "AI connections", level: 1 })).toBeVisible();
  await expect(page.locator(".nav-link.active")).toHaveCount(1);
  await expect(page.locator(".nav-link.active")).toContainText("AI connections");
  await expect(page.locator(".connection-state")).toBeVisible();
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0);
  await expect(page.locator(".required-tag")).toHaveCount(3);
  await expect(page.locator(".optional-tag")).toHaveCount(4);
  await expect(page.locator("#provider-key")).toHaveAttribute("type", "password");
  await expect(page.locator(".model-choice[aria-pressed='true']")).toHaveCount(1);
  const miniCard = page.getByRole("button", { name: /GPT-4o mini/ });
  await expect(miniCard).toContainText("$0.15");
  await expect(miniCard).toContainText("$0.60");
  await expect(miniCard).toContainText("Fastest");
  await expect(page.locator(".scale-bars, .model-scale")).toHaveCount(0);
  await page.getByRole("button", { name: /Anthropic.*4 compatible models/ }).click();
  await expect(page.getByRole("button", { name: /Claude Fable 5/ })).toContainText("$50.00");
  // 4 since gemini-3.1-flash-lite (the model this project's .env actually
  // runs on) was added to the catalog.
  await page.getByRole("button", { name: /Google Gemini.*4 compatible models/ }).click();
  await expect(page.getByRole("button", { name: /Gemini 3.6 Flash/ })).toContainText("$7.50");
  await page.getByRole("button", { name: /OpenAI.*5 compatible models/ }).click();
  const aiLayout = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(aiLayout.scrollWidth).toBeLessThanOrEqual(aiLayout.clientWidth);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  await page.screenshot({ path: path.join(evidenceDirectory, "ai-connections-desktop-en.png"), fullPage: true });
  await page.setViewportSize({ width: 375, height: 812 });
  await expect.poll(() => page.locator(".sidebar").evaluate((element) => element.getBoundingClientRect().right)).toBeLessThanOrEqual(1);
  const aiMobileLayout = await page.evaluate(() => {
    const sidebar = document.querySelector(".sidebar")?.getBoundingClientRect();
    return { client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth, sidebarRight: sidebar?.right ?? 0 };
  });
  expect(aiMobileLayout.scroll).toBeLessThanOrEqual(aiMobileLayout.client);
  expect(aiMobileLayout.sidebarRight).toBeLessThanOrEqual(1);
  await page.screenshot({ path: path.join(evidenceDirectory, "ai-connections-mobile-en.png"), fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });

  await page.getByRole("link", { name: "Data explorer" }).click();
  await page.getByRole("button", { name: "Add invoices or data" }).click();
  await expect(page.getByRole("heading", { name: "Add invoices or source data" })).toBeVisible();
  await page.locator("input[type=file]").first().setInputFiles({
    name: "pos_transactions.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("transaction_id,amount\npreview,10\n"),
  });
  await expect(page.locator(".upload-list")).toContainText("pos_transactions.csv");
  await expect(page.locator(".upload-list")).toContainText("text/csv");
  await expect(page.locator(".upload-list")).toContainText("Modified");
  await page.getByRole("button", { name: "Close", exact: true }).click();

  const desktopLayout = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    activeNavCount: document.querySelectorAll(".nav-link.active").length,
  }));
  expect(desktopLayout.scrollWidth).toBeLessThanOrEqual(desktopLayout.clientWidth);
  expect(desktopLayout.activeNavCount).toBe(1);
  await page.screenshot({ path: path.join(evidenceDirectory, "data-explorer-desktop-en.png"), fullPage: true });

  await page.setViewportSize({ width: 375, height: 812 });
  const viewportWidths = await page.evaluate(() => ({ client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
  expect(viewportWidths.scroll).toBeLessThanOrEqual(viewportWidths.client);
  await page.screenshot({ path: path.join(evidenceDirectory, "data-explorer-mobile-en.png"), fullPage: true });
});
