# Commit Message & Changelog Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErvinasE/commit-message-generator/blob/main/commit_gen_demo.ipynb)

A university course project: an information system that uses generative AI (the Google Gemini API) to

- turn a **git diff** into a commit message in the [Conventional Commits](https://www.conventionalcommits.org/) format, and
- turn a **list of commits** into a changelog grouped into Added, Changed, Fixed, Documentation and Internal.

Two checks sit on top of the model. A **validator** checks every commit message against the format rules and, if it fails, sends the errors back to the model for up to 2 retries. A **coverage check** makes sure the changelog lists every input commit exactly once and invents none.

The whole prototype is one notebook, [`commit_gen_demo.ipynb`](commit_gen_demo.ipynb). To run it, open it in Colab with the badge above, add your Gemini API key as a Colab secret named `GEMINI_API_KEY` (key icon in the left sidebar), and choose Runtime > Run all.

## Repository layout

```
commit_gen_demo.ipynb     the prototype: all code, explanations and results
data/                     (X, y) examples: X = input, y = expected output
  example_1_bugfix/         X_diff.patch, y_expected.txt
  example_2_feature/
  example_3_refactor/
  example_4_docs/
  example_5_multifile/
  changelog_example/        X_commits.json, y_expected_CHANGELOG.md
prompts/
  system_prompt.md          the system instruction the notebook sends to Gemini
  test_prompt.md            a self-contained test prompt, with an AI agent's answer
docs/
  diagram_prompt.md         the prompt used to design the system
  diagrams/                 architecture and sequence diagrams (PlantUML and Mermaid)
```

### `data/`

Every `example_*` folder is one (X, y) pair. `X_diff.patch` is the input diff and `y_expected.txt` is the commit message we expect. `changelog_example/` holds a list of 13 commits (`X_commits.json`) and the expected changelog (`y_expected_CHANGELOG.md`).

The examples are synthetic. They come from a small Python project, `spendlog` (a command-line expense tracker), written for this purpose in a throwaway git repository with 13 commits of different types. Each diff was taken with `git show --format= <hash>`, which leaves the commit message out of the input. Each example is clearly one type of change (bug fix, feature, refactor, docs only, code + tests + config), so the expected answer is unambiguous.

### `prompts/`

`system_prompt.md` is the exact system instruction the notebook reads and sends to Gemini. Its two worked examples are unrelated to the diffs in `data/`.

`test_prompt.md` lets you test the prompt without the notebook. Copy everything above the "Expected output (y)" heading into any AI assistant; it should answer with a commit message. The expected answer and the AI agent's actual answer are recorded under the last two headings.

## Limitations

This is a demonstration, not a benchmark. Five commit examples and one changelog are far too few to measure accuracy, and all of them are synthetic and come from one small project, so real-world commits (often mixed or vague) would be harder. The validator checks format, not meaning: only the commit type is compared automatically. Generating commit messages with AI is not new; what this project adds is the validate-and-retry loop and the changelog coverage check. Finally, the diff is sent to Google's Gemini API, so do not run the notebook on code that must not leave your machine.
