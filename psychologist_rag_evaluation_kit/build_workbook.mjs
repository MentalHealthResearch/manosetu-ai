import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const kitDir = path.resolve("psychologist_rag_evaluation_kit");
const sourceQuestions = path.resolve("mental_health_research", "test_questions.csv");
const outputDir = path.join(kitDir, "outputs", "f82004b3");
const font = "Arial";
const models = ["Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B", "MentalLLaMA-13B"];
const conditions = ["RAG", "No-RAG"];
const metrics = ["Empathy", "Relevance", "Clarity", "Actionability", "Justification", "Helpfulness"];
const raters = ["P01", "P02", "P03", "P04", "P05"];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { cell += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(cell); cell = ""; }
    else if (char === '\n') { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += char;
  }
  if (cell || row.length) { row.push(cell.replace(/\r$/, "")); rows.push(row); }
  const [headers, ...values] = rows;
  return values.filter((values) => values.some((value) => value !== "")).map((values) => Object.fromEntries(headers.map((header, i) => [header, values[i] ?? ""])));
}

function escapeCsv(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csv(rows, fields) {
  return [fields.join(","), ...rows.map((row) => fields.map((field) => escapeCsv(row[field])).join(","))].join("\n") + "\n";
}

function styleTitle(sheet, range, text) {
  sheet.getRange(range).values = [[text]];
  sheet.getRange(range).format = { font: { name: font, size: 14, bold: true, color: "#1F2937" } };
}

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: "#1F4E78",
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
}

function applyBody(sheet, range) {
  sheet.getRange(range).format = { font: { name: font, size: 10, color: "#1F2937" }, verticalAlignment: "center" };
}

const questions = parseCsv(await fs.readFile(sourceQuestions, "utf8")).filter((question) => question.subquestion_id === "1");
if (questions.length !== 40) throw new Error(`Expected 40 primary questions, found ${questions.length}.`);
const responseRows = [];
for (const model of models) for (const condition of conditions) for (const q of questions) {
  responseRows.push({
    answer_id: `${model}__${condition}__Q${String(q.question_id).padStart(2, "0")}`,
    model,
    condition,
    question_id: q.question_id,
    question_text: q.question_text,
    response_text: "",
    rag_corpus_version: condition === "RAG" ? "" : "Not applicable",
    retrieval_top_k: condition === "RAG" ? "" : "0",
    prompt_version: "",
    model_version: "",
    generation_date: "",
    run_notes: "",
  });
}
const answerKey = responseRows.map((row, index) => ({
  blind_answer_id: `A${String(index + 1).padStart(4, "0")}`,
  answer_id: row.answer_id,
  model: row.model,
  condition: row.condition,
  question_id: row.question_id,
}));
const answerMap = new Map(responseRows.map((row) => [row.answer_id, row]));
const ratingRows = [];
for (const rater of raters) for (const key of answerKey) {
  ratingRows.push({
    rater_id: rater,
    blind_answer_id: key.blind_answer_id,
    question_id: key.question_id,
    model: key.model,
    condition: key.condition,
    empathy: "",
    relevance: "",
    clarity: "",
    actionability: "",
    justification: "",
    helpfulness: "",
    feedback_comments: "",
  });
}

await fs.mkdir(path.join(kitDir, "data"), { recursive: true });
await fs.mkdir(path.join(kitDir, "rater_packets"), { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(path.join(kitDir, "data", "questions.csv"), csv(questions, ["question_id", "question_text"]), "utf8");
await fs.writeFile(path.join(kitDir, "data", "response_manifest.csv"), csv(responseRows, Object.keys(responseRows[0])), "utf8");
await fs.writeFile(path.join(kitDir, "data", "ratings_master_template.csv"), csv(ratingRows, Object.keys(ratingRows[0])), "utf8");
await fs.writeFile(path.join(kitDir, "rater_packets", "private_assignment_key_TEMPLATE_ONLY.csv"), csv(answerKey, Object.keys(answerKey[0])), "utf8");
for (let index = 0; index < raters.length; index += 1) {
  const rows = answerKey.map((key) => {
    const answer = answerMap.get(key.answer_id);
    return { rater_id: raters[index], blind_answer_id: key.blind_answer_id, question_id: key.question_id, question_text: answer.question_text, response_text: "", empathy: "", relevance: "", clarity: "", actionability: "", justification: "", helpfulness: "", feedback_comments: "" };
  });
  await fs.writeFile(path.join(kitDir, "rater_packets", `psychologist_${String(index + 1).padStart(2, "0")}_blank_template.csv`), csv(rows, Object.keys(rows[0])), "utf8");
}

const workbook = Workbook.create();
const table5 = workbook.worksheets.add("Table 5");
const protocol = workbook.worksheets.add("Protocol");
const codebook = workbook.worksheets.add("Codebook");
const responses = workbook.worksheets.add("Responses");
const ratings = workbook.worksheets.add("Ratings Master");
const questionsSheet = workbook.worksheets.add("Questions");
for (const sheet of [table5, protocol, codebook, responses, ratings, questionsSheet]) sheet.showGridLines = false;
table5.tabColor = "#1F4E78";

styleTitle(table5, "A2", "Table 5: Mean expert ratings (1-5) across four LLMs in RAG and No-RAG conditions");
table5.getRange("A3").values = [["Final values appear only after all 40 questions have valid ratings from all five psychologists."]];
table5.getRange("A3").format = { font: { name: font, size: 10, italic: true, color: "#4B5563" } };
table5.getRange("A5:I5").values = [["Model", "Cond.", "Empathy", "Relevance", "Clarity", "Action.", "Justif.", "Helpfulness", "Valid scores / cell"]];
styleHeader(table5, "A5:I5");
const summaryRows = [];
for (const model of models) for (const condition of conditions) summaryRows.push([model, condition]);
table5.getRange("A6:B13").values = summaryRows;
for (let row = 6; row <= 13; row += 1) {
  for (let metricCol = 0; metricCol < 6; metricCol += 1) {
    const sourceCol = String.fromCharCode("F".charCodeAt(0) + metricCol);
    const targetCol = String.fromCharCode("C".charCodeAt(0) + metricCol);
    table5.getRange(`${targetCol}${row}`).formulas = [[`=IF(COUNTIFS('Ratings Master'!$D$6:$D$1605,$A${row},'Ratings Master'!$E$6:$E$1605,$B${row},'Ratings Master'!$${sourceCol}$6:$${sourceCol}$1605,\">=1\",'Ratings Master'!$${sourceCol}$6:$${sourceCol}$1605,\"<=5\")=200,AVERAGEIFS('Ratings Master'!$${sourceCol}$6:$${sourceCol}$1605,'Ratings Master'!$D$6:$D$1605,$A${row},'Ratings Master'!$E$6:$E$1605,$B${row}),\"n.a.\")`]];
  }
  table5.getRange(`I${row}`).formulas = [[`=COUNTIFS('Ratings Master'!$D$6:$D$1605,$A${row},'Ratings Master'!$E$6:$E$1605,$B${row},'Ratings Master'!$F$6:$F$1605,\">=1\",'Ratings Master'!$F$6:$F$1605,\"<=5\")`]];
}
applyBody(table5, "A6:I13");
table5.getRange("A5:I13").format.borders = { preset: "insideHorizontal", style: "thin", color: "#D1D5DB" };
table5.getRange("C6:H13").format.numberFormat = "0.00";
table5.getRange("I6:I13").format.numberFormat = "0";
table5.getRange("A17").values = [["Required per metric cell"]];
table5.getRange("B17").values = [[200]];
table5.getRange("D17").values = [["Completed metric cells"]];
table5.getRange("E17").formulas = [["=COUNT(C6:H13)"]];
table5.getRange("A17:E17").format = { fill: "#E8F0F8", font: { name: font, size: 10, bold: true, color: "#1F2937" } };
table5.getRange("A5:I15").format.horizontalAlignment = "center";
table5.getRange("A6:A15").format.horizontalAlignment = "left";
table5.getRange("A2:I2").format.borders = { bottom: { style: "medium", color: "#1F4E78" } };
table5.getRange("A:I").format.columnWidth = 14;
table5.getRange("A:A").format.columnWidth = 22;
table5.getRange("I:I").format.columnWidth = 17;

styleTitle(protocol, "A2", "Psychologist Rating Protocol");
const protocolRows = [
  ["Design", "Rate the same 40 primary questions for four models in RAG and No-RAG conditions. Each psychologist rates all 320 blinded answers."],
  ["RAG control", "Keep prompt, output limit, model version, sampling settings, and safety policy fixed. Only retrieval context changes. Record corpus version and top-k."],
  ["Blinding", "Do not show model, condition, prior scores, or Table 5 values to psychologists. Use the packet builder after responses are complete."],
  ["Scores", "Use whole numbers 1-5 for every criterion. Do not substitute missing scores with zero."],
  ["Completion", "A Table 5 cell needs 200 valid scores: 40 questions x 5 psychologists."],
  ["Data handling", "Keep raw responses, reviewer files, and the private assignment key access-controlled."],
];
protocol.getRange("A4:B9").values = protocolRows;
protocol.getRange("A4:A9").format = { fill: "#E8F0F8", font: { name: font, size: 10, bold: true, color: "#1F2937" }, verticalAlignment: "top" };
protocol.getRange("B4:B9").format = { font: { name: font, size: 10, color: "#1F2937" }, wrapText: true, verticalAlignment: "top" };
protocol.getRange("A4:B9").format.borders = { preset: "all", style: "thin", color: "#D1D5DB" };
protocol.getRange("A:A").format.columnWidth = 20;
protocol.getRange("B:B").format.columnWidth = 100;
protocol.getRange("4:9").format.rowHeight = 35;

styleTitle(codebook, "A2", "Rating Codebook (1-5)");
codebook.getRange("A4:C10").values = [
  ["Criterion", "Assess", "Scale anchor"],
  ["Empathy", "Acknowledges emotion with a respectful, supportive tone.", "1 seriously inadequate; 3 adequate but limited; 5 excellent"],
  ["Relevance", "Directly addresses the specific question and context.", "1 seriously inadequate; 3 adequate but limited; 5 excellent"],
  ["Clarity", "Is understandable, organized, and free of avoidable ambiguity.", "1 seriously inadequate; 3 adequate but limited; 5 excellent"],
  ["Actionability", "Offers feasible next steps appropriate to the situation.", "1 seriously inadequate; 3 adequate but limited; 5 excellent"],
  ["Justification", "Explains suggestions with sound reasoning.", "1 seriously inadequate; 3 adequate but limited; 5 excellent"],
  ["Helpfulness", "Overall expected usefulness considering the full answer.", "1 seriously inadequate; 3 adequate but limited; 5 excellent"],
];
styleHeader(codebook, "A4:C4");
applyBody(codebook, "A5:C10");
codebook.getRange("A4:C10").format.borders = { preset: "all", style: "thin", color: "#D1D5DB" };
codebook.getRange("B5:C10").format.wrapText = true;
codebook.getRange("A:A").format.columnWidth = 18;
codebook.getRange("B:B").format.columnWidth = 62;
codebook.getRange("C:C").format.columnWidth = 45;
codebook.getRange("5:10").format.rowHeight = 34;

styleTitle(responses, "A2", "Response Manifest (research team only)");
responses.getRange("A3").values = [["Enter one generated response for every model-condition-question row. Do not distribute this sheet to psychologists."]];
responses.getRange("A3").format = { font: { name: font, size: 10, italic: true, color: "#4B5563" } };
const responseHeaders = ["Answer ID", "Model", "Condition", "Question ID", "Question text", "Response text", "RAG corpus version", "Top-k", "Prompt version", "Model version", "Generation date", "Run notes"];
responses.getRange("A5:L5").values = [responseHeaders];
styleHeader(responses, "A5:L5");
responses.getRange("A6:L325").values = responseRows.map((row) => [row.answer_id, row.model, row.condition, row.question_id, row.question_text, row.response_text, row.rag_corpus_version, row.retrieval_top_k, row.prompt_version, row.model_version, row.generation_date, row.run_notes]);
applyBody(responses, "A6:L325");
responses.getRange("F6:F325").format.fill = "#FFF2CC";
responses.getRange("G6:K325").format.fill = "#FFF2CC";
responses.getRange("A5:L325").format.wrapText = false;
responses.getRange("A:A").format.columnWidth = 35;
responses.getRange("B:C").format.columnWidth = 18;
responses.getRange("D:D").format.columnWidth = 12;
responses.getRange("E:E").format.columnWidth = 52;
responses.getRange("F:F").format.columnWidth = 70;
responses.getRange("G:L").format.columnWidth = 18;
responses.freezePanes.freezeRows(5);

styleTitle(ratings, "A2", "Ratings Master (research team only)");
ratings.getRange("A3").values = [["Import scored packets here only after joining them to the private assignment key. Yellow cells accept whole-number scores from 1 to 5."]];
ratings.getRange("A3").format = { font: { name: font, size: 10, italic: true, color: "#4B5563" } };
const ratingHeaders = ["Rater ID", "Blind answer ID", "Question ID", "Model", "Condition", ...metrics, "Feedback comments"];
ratings.getRange("A5:L5").values = [ratingHeaders];
styleHeader(ratings, "A5:L5");
ratings.getRange("A6:L1605").values = ratingRows.map((row) => [row.rater_id, row.blind_answer_id, row.question_id, row.model, row.condition, row.empathy, row.relevance, row.clarity, row.actionability, row.justification, row.helpfulness, row.feedback_comments]);
applyBody(ratings, "A6:L1605");
ratings.getRange("F6:K1605").format.fill = "#FFF2CC";
for (const letter of ["F", "G", "H", "I", "J", "K"]) ratings.dataValidations.add({ range: `${letter}6:${letter}1605`, rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 } });
ratings.getRange("A:A").format.columnWidth = 11;
ratings.getRange("B:B").format.columnWidth = 15;
ratings.getRange("C:C").format.columnWidth = 12;
ratings.getRange("D:E").format.columnWidth = 18;
ratings.getRange("F:K").format.columnWidth = 14;
ratings.getRange("L:L").format.columnWidth = 42;
ratings.freezePanes.freezeRows(5);

styleTitle(questionsSheet, "A2", "Shared Primary Question Set");
questionsSheet.getRange("A3").values = [["The 40 primary prompts, aligned to the existing psychologist-feedback file. Use the same questions for every model and condition."]];
questionsSheet.getRange("A3").format = { font: { name: font, size: 10, italic: true, color: "#4B5563" } };
questionsSheet.getRange("A5:B5").values = [["Question ID", "Question text"]];
styleHeader(questionsSheet, "A5:B5");
questionsSheet.getRange("A6:B45").values = questions.map((q) => [Number(q.question_id), q.question_text]);
applyBody(questionsSheet, "A6:B45");
questionsSheet.getRange("B6:B45").format.wrapText = true;
questionsSheet.getRange("A:A").format.columnWidth = 14;
questionsSheet.getRange("B:B").format.columnWidth = 100;
questionsSheet.getRange("6:45").format.rowHeight = 32;
questionsSheet.freezePanes.freezeRows(5);

workbook.recalculate();
const inspect = await workbook.inspect({ kind: "table", range: "Table 5!A2:I17", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 10 });
console.log(inspect.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 50 }, summary: "final formula error scan" });
console.log(errors.ndjson);
for (const [sheetName, range] of [["Table 5", "A1:I18"], ["Protocol", "A1:B10"], ["Codebook", "A1:C11"], ["Responses", "A1:L12"], ["Ratings Master", "A1:L12"], ["Questions", "A1:B12"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(path.join(outputDir, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(outputDir, "LLM_RAG_NoRAG_psychologist_evaluation.xlsx"));
console.log(`Created workbook and new evaluation files in ${kitDir}`);
