export type TerminalLineType =
  | "header"
  | "decompose"
  | "assign"
  | "execute"
  | "complete"
  | "verify_pass"
  | "verify_fail"
  | "trust"
  | "reassign"
  | "result"
  | "blank";

export interface DemoLine {
  text: string;
  type: TerminalLineType;
}

export const demoLines: DemoLine[] = [
  { text: "==================================================", type: "header" },
  { text: "  delegato — Research Pipeline Demo", type: "header" },
  { text: "==================================================", type: "header" },
  { text: "", type: "blank" },
  { text: "[DECOMPOSE]  Breaking task into 3 sub-tasks...", type: "decompose" },
  { text: "[ASSIGN]  task → searcher", type: "assign" },
  { text: "[EXECUTE]  searcher running...", type: "execute" },
  { text: "[COMPLETE]  searcher done", type: "complete" },
  { text: "[VERIFY]  regex: PASS (drug discovery|AI.+pharma|molecule)", type: "verify_pass" },
  { text: "[TRUST]  searcher.web_search: 0.50 → 0.55", type: "trust" },
  { text: "[ASSIGN]  task → analyzer", type: "assign" },
  { text: "[EXECUTE]  analyzer running...", type: "execute" },
  { text: "[COMPLETE]  analyzer done", type: "complete" },
  { text: "[VERIFY]  regex: PASS (confidence)", type: "verify_pass" },
  { text: "[TRUST]  analyzer.data_analysis: 0.50 → 0.55", type: "trust" },
  { text: "[ASSIGN]  task → synthesizer", type: "assign" },
  { text: "[EXECUTE]  synthesizer running...", type: "execute" },
  { text: "[COMPLETE]  synthesizer done", type: "complete" },
  { text: "[VERIFY]  llm_judge: FAIL (Only 2 examples found, need at least 3)", type: "verify_fail" },
  { text: "[TRUST]  synthesizer.summarization: 0.50 → 0.42", type: "trust" },
  { text: "[REASSIGN]  → synthesizer (retry 1/2)", type: "reassign" },
  { text: "[EXECUTE]  synthesizer running...", type: "execute" },
  { text: "[COMPLETE]  synthesizer done", type: "complete" },
  { text: "[VERIFY]  llm_judge: PASS (3 examples, 487 words, good quality)", type: "verify_pass" },
  { text: "[TRUST]  synthesizer.summarization: 0.42 → 0.49", type: "trust" },
  { text: "", type: "blank" },
  { text: "==================================================", type: "header" },
  { text: "  RESULT: SUCCESS", type: "result" },
  { text: "  Total time: 0.3s | Cost: $0.037 | Reassignments: 1", type: "result" },
  { text: "==================================================", type: "header" },
];
