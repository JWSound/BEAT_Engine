# Collaboration rules / Zusammenarbeit (Claude + ChatGPT, owner: repository owner)

English first, Deutsch unten. Either language is fine in issues, PR descriptions and review
comments (German preferred for discussion so the whole team can follow). Code, code comments
and commit messages stay in English, as in the rest of the repository (see AGENTS.md).

## English

**Roles.** The repository owner decides and merges. Two AI agents contribute:
Claude (works with the owner's Mac test runner and rented GPUs) and ChatGPT (independent
review and its own components). Neither agent merges into `main` or `memory-fork`.

**Branches.**
- `main` – unchanged mirror of upstream. Never committed to.
- `memory-fork` – the fork actually used by Boundary Lab. Changes only via pull request
  approved by the owner.
- `hmatrix-dev` – experimental H-matrix work (WORK IN PROGRESS, not used by any solve path).
- Agent work: `claude/<topic>` and `chatgpt/<topic>`. Each agent pushes only to its own prefix.

**Ownership.** The author of a change owns it. The other agent does not edit it; it proposes
changes in a PR review (GitHub "suggestion" blocks or a patch attached to the comment). The
owner of the code applies, rejects or discusses the proposal on its own branch.

| Area | Owner |
|---|---|
| `src/beat_engine/julia_local/hmatrix_dev/**`, `src/BeatEngineFmm.jl`, `tests/fmm_*`, `docs/FMM_PLAN.md` | Claude |
| `src/BeatEngineCpuBlockSampler.jl`, `tests/cpu_block_sampler_tests.jl`, `docs/CpuGalerkinBlockSampler.md` | ChatGPT |
| Split dense refinement guards (commit 9d71960) | ChatGPT (author), merged by the owner |
| Everything else in `memory-fork` | the owner; changes by PR only |

**Tasks.** One GitHub issue per task, labels `claude`, `chatgpt`, `review`, `bug`, `measurement`.
The owner assigns; agents may propose new issues.

**What a PR must contain.**
1. What changed and why; which solve paths it can affect (or "none – opt-in / not called").
2. Tests run, with command lines and pass counts; new regression tests for every bug fix.
3. For anything numerical: accuracy against a defined reference (the Float64 dense reference
   of a real project where possible), stated as the maximum complex relative error per output
   quantity, not only an L2 norm. Float32 dense results are NOT a valid reference at high
   frequency (they differ from Float64 by up to 2 % in boundary pressure at 20 kHz).
4. Performance claims follow docs/Benchmarking.md (3 interleaved runs, median and range).
5. Numerical changes in their own PR, never mixed with refactoring (AGENTS.md).

**Reviews.** Every PR into `memory-fork` gets one review by the other agent before the owner
merges. A review states: verdict (approve / changes requested / comment), concrete findings with
file:line, and whether the finding was reproduced.

**Ideas.** Both agents are expected to propose ideas to each other, not only to review: alternative
algorithms, measurements worth running, simplifications, risks seen in the other's plan.
An idea is a GitHub issue with label `idea` (or a Discussion, if enabled) containing:
problem it addresses; the idea in a few sentences; expected gain (accuracy, memory, time, effort);
how it would be tested (reference, model, frequency); rough cost (hours of work, GPU rental).
The agent that owns the affected area answers within its next work session: adopt (opens a task
issue), reject (with the reason), or ask a question. The owner of the repository has the final word.

## Deutsch

**Rollen.** Der Repository-Besitzer entscheidet und merged. Zwei KI-Agenten arbeiten mit:
Claude (mit dem Mac-Testrunner des Besitzers und gemieteten GPUs) und ChatGPT (unabhängige
Prüfung und eigene Komponenten). Kein Agent merged in `main` oder `memory-fork`.

**Branches.**
- `main` – unveränderter Spiegel von Upstream. Keine Commits.
- `memory-fork` – der von Boundary Lab tatsächlich genutzte Fork. Änderungen nur per Pull
  Request mit Freigabe des Besitzers.
- `hmatrix-dev` – experimentelle H-Matrix-Arbeit (WORK IN PROGRESS, von keinem Lösungspfad genutzt).
- Agentenarbeit: `claude/<thema>` und `chatgpt/<thema>`. Jeder Agent pusht nur auf sein Präfix.

**Zuständigkeit.** Wer eine Änderung schreibt, ist dafür zuständig. Der andere Agent ändert sie
nicht, sondern macht Vorschläge im PR-Review (GitHub-"Suggestion"-Blöcke oder ein Patch im
Kommentar). Der Zuständige übernimmt, verwirft oder diskutiert den Vorschlag auf seinem Branch.
Zuständigkeitstabelle: siehe englischer Teil.

**Aufgaben.** Ein GitHub-Issue pro Aufgabe, Labels `claude`, `chatgpt`, `review`, `bug`,
`measurement`. Der Besitzer verteilt; Agenten dürfen neue Issues vorschlagen.

**Inhalt eines PR.**
1. Was und warum; welche Lösungspfade betroffen sein können (oder „keine – opt-in / nicht aufgerufen“).
2. Ausgeführte Tests mit Befehlszeilen und Anzahl bestandener Prüfungen; Regressionstest für jeden Bugfix.
3. Numerische Änderungen: Genauigkeit gegen eine definierte Referenz (wo möglich die Float64-
   Dichtreferenz eines echten Projekts), als maximaler komplexer relativer Fehler je Ausgabegröße,
   nicht nur als L2-Norm. Float32-Dichtergebnisse sind bei hohen Frequenzen KEINE gültige Referenz
   (bis zu 2 % Abweichung im Randdruck bei 20 kHz gegenüber Float64).
4. Performance-Aussagen nach docs/Benchmarking.md (3 abwechselnde Läufe, Median und Spannweite).
5. Numerische Änderungen in einem eigenen PR, nie mit Refactoring gemischt (AGENTS.md).

**Reviews.** Jeder PR nach `memory-fork` bekommt vor dem Merge ein Review des anderen Agenten.
Ein Review nennt: Urteil (approve / changes requested / comment), konkrete Befunde mit Datei:Zeile,
und ob der Befund reproduziert wurde.

**Ideen.** Beide Agenten sollen einander auch Ideen geben, nicht nur prüfen: alternative Verfahren,
lohnende Messungen, Vereinfachungen, Risiken im Plan des anderen. Eine Idee ist ein GitHub-Issue mit
dem Label `idea` (oder eine Discussion, falls aktiviert) mit: Problem; die Idee in wenigen Sätzen;
erwarteter Nutzen (Genauigkeit, Speicher, Zeit, Aufwand); wie sie geprüft würde (Referenz, Modell,
Frequenz); grober Aufwand (Arbeitsstunden, GPU-Miete). Der für den Bereich zuständige Agent antwortet in
seiner nächsten Arbeitssitzung: übernehmen (eröffnet ein Aufgaben-Issue), ablehnen (mit Begründung) oder
nachfragen. Das letzte Wort hat der Repository-Besitzer.
