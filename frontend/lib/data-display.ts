import type { SourceSummary } from "@/lib/types";

export type FileCandidate = { file: File; relativePath: string };
export type UploadFileState = "new" | "existing" | "identical" | "duplicate_selection" | "unsupported";
export type UploadAssessment = FileCandidate & { source?: string; state: UploadFileState; detail: string };

const sourceFiles: Record<string, string> = {
  pos_transactions: "pos", menu_items: "menu", foot_traffic: "traffic", staff_shifts: "staff",
  inventory_weekly: "inventory", customer_reviews: "reviews",
};

const labels: Record<string, string> = {
  pos: "POS", menu: "Menu", traffic: "Traffic", staff: "Staff",
  inventory: "Inventory", emails: "Emails", reviews: "Reviews",
  sku: "Item code", item_en: "Item", item_ar: "Arabic name", category: "Category",
  price_sar: "Price (SAR)", unit_cost_sar: "Unit cost (SAR)", is_iced: "Iced",
  door_count: "Visitors", date: "Date", hour: "Hour", sender: "Sender", subject: "Subject",
  confidence: "Confidence", employee_id: "Employee", shift_start: "Shift starts", shift_end: "Shift ends",
  transaction_id: "Transaction", quantity: "Quantity", payment_method: "Payment method",
  week_starting: "Week starting", on_hand_qty: "On hand", waste_qty: "Waste",
};

const itemTranslations: Record<string, { en: string, ar: string }> = {
  "butter croissant": { en: "Butter Croissant", ar: "كرواسون" },
  "كرواسون": { en: "Butter Croissant", ar: "كرواسون" },
  "cinnamon roll": { en: "Cinnamon Roll", ar: "سينامون رول" },
  "سينامون رول": { en: "Cinnamon Roll", ar: "سينامون رول" },
  "date cake": { en: "Date Cake", ar: "كيكة التمر" },
  "كيكة التمر": { en: "Date Cake", ar: "كيكة التمر" },
  "cheesecake slice": { en: "Cheesecake Slice", ar: "تشيز كيك" },
  "تشيز كيك": { en: "Cheesecake Slice", ar: "تشيز كيك" },
  "chicken sandwich": { en: "Chicken Sandwich", ar: "ساندويتش دجاج" },
  "ساندويتش دجاج": { en: "Chicken Sandwich", ar: "ساندويتش دجاج" },
  "halloumi toast": { en: "Halloumi Toast", ar: "توست حلومي" },
  "توست حلومي": { en: "Halloumi Toast", ar: "توست حلومي" },
  "spanish latte": { en: "Spanish Latte", ar: "سبانيش لاتيه" },
  "سبانيش لاتيه": { en: "Spanish Latte", ar: "سبانيش لاتيه" },
  "flat white": { en: "Flat White", ar: "فلات وايت" },
  "فلات وايت": { en: "Flat White", ar: "فلات وايت" },
  "cortado": { en: "Cortado", ar: "كورتادو" },
  "كورتادو": { en: "Cortado", ar: "كورتادو" },
  "v60 filter": { en: "V60 Filter", ar: "في60" },
  "في60": { en: "V60 Filter", ar: "في60" },
  "karak tea": { en: "Karak Tea", ar: "كرك" },
  "كرك": { en: "Karak Tea", ar: "كرك" },
  "saffron latte": { en: "Saffron Latte", ar: "لاتيه زعفران" },
  "لاتيه زعفران": { en: "Saffron Latte", ar: "لاتيه زعفران" },
  "hot chocolate": { en: "Hot Chocolate", ar: "هوت شوكليت" },
  "هوت شوكليت": { en: "Hot Chocolate", ar: "هوت شوكليت" },
  "iced spanish latte": { en: "Iced Spanish Latte", ar: "آيس سبانيش لاتيه" },
  "آيس سبانيش لاتيه": { en: "Iced Spanish Latte", ar: "آيس سبانيش لاتيه" },
  "iced americano": { en: "Iced Americano", ar: "آيس أمريكانو" },
  "آيس أمريكانو": { en: "Iced Americano", ar: "آيس أمريكانو" },
  "iced latte": { en: "Iced Latte", ar: "آيس لاتيه" },
  "آيس لاتيه": { en: "Iced Latte", ar: "آيس لاتيه" },
  "cold brew": { en: "Cold Brew", ar: "كولد برو" },
  "كولد برو": { en: "Cold Brew", ar: "كولد برو" },
  "matcha latte": { en: "Matcha Latte", ar: "ماتشا لاتيه" },
  "ماتشا لاتيه": { en: "Matcha Latte", ar: "ماتشا لاتيه" },
  "mojito passion": { en: "Mojito Passion", ar: "موهيتو باشن" },
  "موهيتو باشن": { en: "Mojito Passion", ar: "موهيتو باشن" },
  "takeaway": { en: "Takeaway", ar: "سفري" },
  "سفري": { en: "Takeaway", ar: "سفري" },
  "dine in": { en: "Dine-in", ar: "محلي" },
  "dine_in": { en: "Dine-in", ar: "محلي" },
  "محلي": { en: "Dine-in", ar: "محلي" },
  "mada": { en: "Mada", ar: "مدى" },
  "مدى": { en: "Mada", ar: "مدى" },
  "apple pay": { en: "Apple Pay", ar: "أبل باي" },
  "أبل باي": { en: "Apple Pay", ar: "أبل باي" },
  "cash": { en: "Cash", ar: "كاش" },
  "كاش": { en: "Cash", ar: "كاش" },
  "visa": { en: "Visa", ar: "فيزا" },
  "فيزا": { en: "Visa", ar: "فيزا" },
  "delivery": { en: "Delivery", ar: "توصيل" },
  "توصيل": { en: "Delivery", ar: "توصيل" },
  "hot coffee": { en: "Hot Coffee", ar: "قهوة ساخنة" },
  "قهوة ساخنة": { en: "Hot Coffee", ar: "قهوة ساخنة" },
  "hot other": { en: "Hot Other", ar: "مشروبات ساخنة أخرى" },
  "مشروبات ساخنة أخرى": { en: "Hot Other", ar: "مشروبات ساخنة أخرى" },
  "iced coffee": { en: "Iced Coffee", ar: "قهوة باردة" },
  "قهوة باردة": { en: "Iced Coffee", ar: "قهوة باردة" },
  "iced other": { en: "Iced Other", ar: "مشروبات باردة أخرى" },
  "مشروبات باردة أخرى": { en: "Iced Other", ar: "مشروبات باردة أخرى" },
  "food": { en: "Food", ar: "مأكولات" },
  "مأكولات": { en: "Food", ar: "مأكولات" },
  "senior_barista": { en: "Senior Barista", ar: "باريستا أول" },
  "senior barista": { en: "Senior Barista", ar: "باريستا أول" },
  "باريستا أول": { en: "Senior Barista", ar: "باريستا أول" },
  "barista": { en: "Barista", ar: "باريستا" },
  "باريستا": { en: "Barista", ar: "باريستا" },
  "cashier": { en: "Cashier", ar: "كاشير" },
  "كاشير": { en: "Cashier", ar: "كاشير" },
  "hussain al-nasser": { en: "Hussain Al-Nasser", ar: "حسين الناصر" },
  "حسين الناصر": { en: "Hussain Al-Nasser", ar: "حسين الناصر" },
  "ali al-mutairi": { en: "Ali Al-Mutairi", ar: "علي المطيري" },
  "علي المطيري": { en: "Ali Al-Mutairi", ar: "علي المطيري" },
  "zahra al-hassan": { en: "Zahra Al-Hassan", ar: "زهراء الحسن" },
  "زهراء الحسن": { en: "Zahra Al-Hassan", ar: "زهراء الحسن" },
  "fatimah al-awami": { en: "Fatimah Al-Awami", ar: "فاطمة العوامي" },
  "فاطمة العوامي": { en: "Fatimah Al-Awami", ar: "فاطمة العوامي" },
  "mohammed al-qadeeh": { en: "Mohammed Al-Qadeeh", ar: "محمد القديح" },
  "محمد القديح": { en: "Mohammed Al-Qadeeh", ar: "محمد القديح" },
  "sara al-zahrani": { en: "Sara Al-Zahrani", ar: "سارة الزهراني" },
  "سارة الزهراني": { en: "Sara Al-Zahrani", ar: "سارة الزهراني" },
  "yousef al-dossary": { en: "Yousef Al-Dossary", ar: "يوسف الدوسري" },
  "يوسف الدوسري": { en: "Yousef Al-Dossary", ar: "يوسف الدوسري" },
  "noor al-sadah": { en: "Noor Al-Sadah", ar: "نور السادة" },
  "نور السادة": { en: "Noor Al-Sadah", ar: "نور السادة" },
  "waited 25 minutes on friday evening, too crowded": { en: "Waited 25 minutes on Friday evening, too crowded", ar: "انتظرت 25 دقيقة مساء الجمعة، زحمة جداً" },
  "parking is difficult": { en: "Parking is difficult", ar: "المواقف صعبة" },
  "used to be my favourite v60 spot, something changed": { en: "Used to be my favourite V60 spot, something changed", ar: "كان مكاني المفضل للـ V60، تغير شيء ما" },
  "karak here is unreal": { en: "Karak here is unreal", ar: "الكرك هنا خرافي" },
  "best spanish latte in saihat honestly": { en: "Best Spanish latte in Saihat honestly", ar: "أفضل سبانيش لاتيه في سيهات بصراحة" },
  "only one barista during rush, queue outside": { en: "Only one barista during rush, queue outside", ar: "باريستا واحد فقط وقت الزحمة، طابور في الخارج" },
  "a bit expensive for the size": { en: "A bit expensive for the size", ar: "غالي قليلاً بالنسبة للحجم" },
  "perfect spot after taraweeh": { en: "Perfect spot after taraweeh", ar: "مكان مثالي بعد التراويح" },
  "great vibe for working, fast wifi": { en: "Great vibe for working, fast wifi", ar: "أجواء رائعة للعمل، واي فاي سريع" },
  "friday nights are chaos, waited forever": { en: "Friday nights are chaos, waited forever", ar: "ليالي الجمعة فوضى، انتظرت طويلاً" },
  "filter coffee was bitter and cold, not like before": { en: "Filter coffee was bitter and cold, not like before", ar: "قهوة الفلتر كانت مرة وباردة، ليست كما كانت" },
  "music too loud": { en: "Music too loud", ar: "الموسيقى صاخبة جداً" },
  "prices went up": { en: "Prices went up", ar: "الأسعار ارتفعت" },
  "staff are so friendly": { en: "Staff are so friendly", ar: "طاقم العمل ودودون جداً" },
  "v60 quality dropped recently": { en: "V60 quality dropped recently", ar: "جودة الـ V60 انخفضت مؤخراً" },
  "clean, quiet, good music": { en: "Clean, quiet, good music", ar: "نظيف، هادئ، موسيقى جيدة" },
  "pour over came out watery": { en: "Pour over came out watery", ar: "القهوة المقطرة جاءت مائية" },
  "cold brew is smooth, will come back": { en: "Cold brew is smooth, will come back", ar: "الكولد برو سلس، سأعود مجدداً" },
  "الانتظار طويل يوم الجمعة": { en: "Wait is long on Friday", ar: "الانتظار طويل يوم الجمعة" },
  "كيكة التمر لذيذة جدا": { en: "Date cake is very delicious", ar: "كيكة التمر لذيذة جدا" },
  "الفلتر ما عاد نفس الطعم": { en: "Filter coffee doesn't taste the same anymore", ar: "الفلتر ما عاد نفس الطعم" },
  "طلبت وانتظرت نص ساعة": { en: "Ordered and waited half an hour", ar: "طلبت وانتظرت نص ساعة" },
  "الماتشا حلوة مررة": { en: "Matcha is very sweet", ar: "الماتشا حلوة مررة" },
  "المكان هادي والقهوة ممتازة": { en: "Place is quiet and coffee is excellent", ar: "المكان هادي والقهوة ممتازة" },
  "weekly milk delivery confirmation": { en: "Weekly milk delivery confirmation", ar: "تأكيد توصيل الحليب الأسبوعي" },
  "green coffee lot arrival — ethiopia guji": { en: "Green coffee lot arrival — Ethiopia Guji", ar: "وصول دفعة القهوة الخضراء - إثيوبيا قوجي" },
  "ramadan delivery schedule change": { en: "Ramadan delivery schedule change", ar: "تغيير جدول التوصيل في رمضان" },
  "ramadan night market — vendor slots": { en: "Ramadan night market — vendor slots", ar: "سوق رمضان الليلي - مقاعد البائعين" },
  "price notice — q2 green coffee": { en: "Price notice — Q2 green coffee", ar: "إشعار الأسعار - القهوة الخضراء الربع الثاني" },
  "ceremonial matcha — sample kit": { en: "Ceremonial matcha — sample kit", ar: "ماتشا احتفالية - عينة" },
  "important: price increase effective 1 may": { en: "IMPORTANT: price increase effective 1 May", ar: "هام: زيادة الأسعار اعتباراً من 1 مايو" },
  "scheduled maintenance — reporting module": { en: "Scheduled maintenance — reporting module", ar: "صيانة مجدولة - نظام التقارير" },
  "...eduled maintenance — reporting module": { en: "...eduled maintenance — reporting module", ar: "صيانة مجدولة - نظام التقارير" },
  "cup and lid quotation": { en: "Cup and lid quotation", ar: "تسعيرة الأكواب والأغطية" },
  "delivery delay — 8 to 10 june": { en: "Delivery delay — 8 to 10 June", ar: "تأخير التوصيل - 8 إلى 10 يونيو" },
  "summer cold brew concentrate": { en: "Summer cold brew concentrate", ar: "مُركّز الكولد برو الصيفي" },
  "noise": { en: "Noise", ar: "إزعاج" },
  "product offer": { en: "Product offer", ar: "عرض منتج" },
  "product_offer": { en: "Product offer", ar: "عرض منتج" },
  "event": { en: "Event", ar: "فعالية" },
  "price change": { en: "Price change", ar: "تغيير سعر" },
  "price_change": { en: "Price change", ar: "تغيير سعر" },
  "maintenance": { en: "Maintenance", ar: "صيانة" },
  "quote": { en: "Quote", ar: "تسعيرة" },
  "delivery delay": { en: "Delivery delay", ar: "تأخير توصيل" },
  "delivery_delay": { en: "Delivery delay", ar: "تأخير توصيل" },
  "roasted price": { en: "Roasted price", ar: "سعر التحميص" },
  "roasted_price": { en: "Roasted price", ar: "سعر التحميص" },
  "house blend": { en: "House blend", ar: "مزيج المحمصة" },
  "house_blend": { en: "House blend", ar: "مزيج المحمصة" },
  "full-fat milk": { en: "Full-fat milk", ar: "حليب كامل الدسم" },
  "full_fat_milk": { en: "Full-fat milk", ar: "حليب كامل الدسم" },
  "barista oat": { en: "Barista oat", ar: "شوفان باريستا" },
  "barista_oat": { en: "Barista oat", ar: "شوفان باريستا" },
  "deterministic fallback": { en: "Deterministic fallback", ar: "استخراج احتياطي" },
  "deterministic_fallback": { en: "Deterministic fallback", ar: "استخراج احتياطي" },
  "google": { en: "Google", ar: "جوجل" },
  "instagram": { en: "Instagram", ar: "انستغرام" },
  "talabat": { en: "Talabat", ar: "طلبات" },
  "positive": { en: "Positive", ar: "إيجابي" },
  "negative": { en: "Negative", ar: "سلبي" },
  "neutral": { en: "Neutral", ar: "محايد" },
};

const labelsAr: Record<string, string> = {
  pos: "الفواتير", menu: "قائمة الطعام", traffic: "الزوار", staff: "الموظفين",
  inventory: "المخزون", emails: "البريد الإلكتروني", reviews: "التقييمات",
  item_en: "المنتج", item_ar: "الاسم بالعربي", category: "التصنيف",
  price_sar: "السعر (ريال)", unit_cost_sar: "التكلفة (ريال)", is_iced: "مثلج",
  date: "التاريخ", hour: "الساعة",
  door_count: "الزوار", sender: "المرسل", subject: "الموضوع",
  confidence: "الموثوقية", employee_id: "رقم الموظف", shift_start: "بداية الوردية", shift_end: "نهاية الوردية",
  hourly_rate_sar: "معدل الأجر بالساعة (ريال)", hours: "ساعات العمل", name: "الاسم",
  quantity: "الكمية", payment_method: "طريقة الدفع",
  week_starting: "بداية الأسبوع", on_hand_qty: "المتوفر", waste_qty: "المهدر",
  total_loss_sar: "إجمالي الخسارة (ريال)", total_profit_sar: "إجمالي الربح (ريال)",
  unit_price_sar: "سعر الوحدة (ريال)", role: "المنصب",
  rating: "التقييم", channel: "القناة", comment: "التعليق", sentiment: "الانطباع",
  entity_or_ingredient: "المكون أو المنتج", extraction_mode: "طريقة الاستخراج",
  old_price: "السعر القديم", new_price: "السعر الجديد", language: "اللغة", text: "النص", source: "المصدر",
  units_ordered: "الوحدات المطلوبة", units_sold: "المباع", units_wasted: "المهدر",
  week_starting_raw: "بداية الأسبوع (أصلي)", line_total_sar: "الإجمالي (ريال)", discount_sar: "الخصم (ريال)",
  item_name: "اسم المنتج"
};

const priorities: Record<string, string[]> = {
  menu: ["item_en", "item_ar", "sku", "category", "price_sar", "unit_cost_sar", "is_iced"],
  traffic: ["date", "hour", "door_count"],
  emails: ["date", "sender", "subject", "category", "entity_or_ingredient", "confidence"],
  pos: ["timestamp", "transaction_id", "sku", "quantity", "unit_price_sar", "payment_method"],
  staff: ["date", "employee_id", "role", "shift_start", "shift_end"],
  inventory: ["week_starting", "sku", "on_hand_qty", "waste_qty", "units_ordered", "units_sold", "units_wasted", "total_loss_sar", "total_profit_sar"],
  reviews: ["date", "rating", "channel", "comment", "sentiment"],
};

export type HumanRow = Record<string, unknown> & { __recordId: string; __rowNumber: number };

export function flattenRows(items: Array<Record<string, unknown>>): HumanRow[] {
  return items.map((row, index) => {
    const nested = row.data && typeof row.data === "object" && !Array.isArray(row.data) ? row.data as Record<string, unknown> : row;
    const calculated: Record<string, unknown> = {};
    const unitsWasted = Number(nested.units_wasted ?? nested.waste_qty ?? 0);
    const unitCost = Number(nested.unit_cost_sar ?? 0);
    if (unitsWasted > 0 && unitCost > 0) calculated.total_loss_sar = unitsWasted * unitCost;
    const unitsSold = Number(nested.units_sold ?? 0);
    const price = Number(nested.price_sar ?? nested.unit_price_sar ?? 0);
    if (unitsSold > 0 && price > 0 && unitCost > 0) calculated.total_profit_sar = unitsSold * (price - unitCost);
    return { ...nested, ...calculated, __recordId: String(row.record_id ?? row.id ?? index), __rowNumber: index + 1 };
  });
}

export function visibleColumns(rows: HumanRow[], sourceId: string) {
  const keys = new Set<string>();
  rows.slice(0, 20).forEach((row) => Object.entries(row).forEach(([key, value]) => {
    if (!key.startsWith("__") && value !== null && value !== undefined && !["evidence_text", "facts", "email_file"].includes(key)) keys.add(key);
  }));
  const preferred = priorities[sourceId] ?? [];
  return [...preferred.filter((key) => keys.has(key)), ...Array.from(keys).filter((key) => !preferred.includes(key))].slice(0, 10);
}

export function humanLabel(key: string, locale: "ar" | "en" = "en") {
  if (locale === "ar" && labelsAr[key]) return labelsAr[key];
  if (locale === "ar") return key.replace(/_/g, " ");
  return labels[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function humanValue(value: unknown, locale: "ar" | "en") {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? (locale === "ar" ? "نعم" : "Yes") : (locale === "ar" ? "لا" : "No");
  if (Array.isArray(value)) return value.map((item) => typeof item === "object" ? JSON.stringify(item) : String(item)).join(" · ");
  if (typeof value === "object") return Object.entries(value as Record<string, unknown>).map(([key, item]) => `${humanLabel(key, locale)}: ${String(item ?? "—")}`).join(" · ");
  if (typeof value === "number") return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
  if (typeof value === "string") {
    const lower = value.toLowerCase().trim();
    if (itemTranslations[lower]) return locale === "ar" ? itemTranslations[lower].ar : itemTranslations[lower].en;
  }
  return String(value).replace(/_/g, " ");
}

function sourceFor(candidate: FileCandidate) {
  const stem = candidate.file.name.replace(/\.[^.]+$/, "").toLowerCase();
  if (candidate.file.name.toLowerCase().endsWith(".txt")) return "emails";
  return sourceFiles[stem];
}

const normalizedText = (value: string) => value.replace(/\r\n/g, "\n").trim().replace(/\s+/g, " ");

export async function assessFiles(candidates: FileCandidate[], sources: SourceSummary[], emailRows: Array<Record<string, unknown>>) {
  const seen = new Set<string>();
  const emails = flattenRows(emailRows);
  const results: UploadAssessment[] = [];
  for (const candidate of candidates) {
    const source = sourceFor(candidate);
    if (!source) {
      results.push({ ...candidate, state: "unsupported", detail: "This filename is not a registered data source." });
      continue;
    }
    const bytes = await candidate.file.arrayBuffer();
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))).map((byte) => byte.toString(16).padStart(2, "0")).join("");
    if (seen.has(digest)) {
      results.push({ ...candidate, source, state: "duplicate_selection", detail: "The same content is already selected in this batch." });
      continue;
    }
    seen.add(digest);
    if (source !== "emails") {
      const exists = Boolean(sources.find((item) => item.id === source && (item.accepted_rows ?? item.raw_rows ?? 0) > 0));
      results.push({ ...candidate, source, state: exists ? "existing" : "new", detail: exists ? "This registered source already has data and can be replaced." : "This is a new registered source file." });
      continue;
    }
    const text = normalizedText(await candidate.file.text());
    const identical = emails.some((email) => {
      const evidence = normalizedText(String(email.evidence_text ?? ""));
      const facts = Array.isArray(email.facts) ? normalizedText(email.facts.join("\n")) : "";
      return Boolean(text && (text.includes(evidence) || evidence.includes(text) || text.includes(facts) || facts.includes(text)));
    });
    if (identical) {
      results.push({ ...candidate, source, state: "identical", detail: "The same email content is already registered; no write is needed." });
      continue;
    }
    const dateFromName = candidate.file.name.match(/^\d{4}-\d{2}-\d{2}/)?.[0];
    const sameSlot = Boolean(dateFromName && emails.some((email) => email.date === dateFromName));
    results.push({ ...candidate, source, state: sameSlot ? "existing" : "new", detail: sameSlot ? "An email for this dated slot already exists and can be replaced." : "This email is new." });
  }
  return results;
}
