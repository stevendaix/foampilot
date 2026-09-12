# Making a CFD Tool Adoptable: Lessons from 6 Months of foampilot on GitHub

*You've spent 6 months building a wonderful tool. It automates 90% of your CFD workflow. You share it on GitHub. Six months later: 3 stars, 0 issues, 0 contributors. The problem? A scientific open-source project lives not only by its publication — it lives by its tool.*

---

## Introduction: The Invisible Tool

Storytelling: You've spent 6 months building a wonderful tool. It automates 90% of your CFD workflow. You share it on GitHub. Six months later: 3 stars, 0 issues, 0 contributors.

**The problem**: A scientific open-source project is a product with two faces:
1. The technical face (code, algorithms, rigor)
2. The communication face (README, examples, documentation, demo)

Most researchers excel at the first. Few invest enough in the second.

**Promise**: The 5 lessons I learned from foampilot to make a tool adoptable.

---

## Lesson 1: The README as a Promise, Not Documentation

**Lesson 1**: The README must answer 3 questions in 30 seconds:
1. What is it?
2. What is it for?
3. Why is it better than the alternative?

Example from foampilot:
- Title + emoji
- Elevator pitch
- Features with action verbs
- "What foampilot is not" section to avoid misunderstandings

**Error to avoid**: A 200-line README that explains architecture before explaining usage.

---

## Lesson 2: Multilingual Documentation as a Signal

If your tool is used globally, translate the storefront.

- OpenFOAM is global, but many users aren't English-speaking
- A README in 3 languages (EN/FR/ZH) sends a message: *"this project is made for you"*
- Translation doesn't have to be perfect — it has to be understandable

---

## Lesson 3: Examples as Sales Arguments

**Lesson 3**: Don't say your tool is powerful. Show it.

- The `examples/` directory is your best argument
- Each example answers a question: "can it do this?"
- A complete example > 10 pages of documentation
- Screenshots and visual results are proofs

**Advice**: For each example, add a `README.md` explaining the case, parameters, and expected results.

---

## Lesson 4: Technical Documentation as Project Memory

**Lesson 4**: An open-source project has two audiences: users and contributors.

- User documentation: tutorials, guides, FAQ
- Contributor documentation: architecture, conventions, dev workflow
- MkDocs or Sphinx for structure
- Cross-links between pages

---

## Lesson 5: Tests as Proof of Maturity

**Lesson 5**: In the scientific world, reproducibility is king.

- A project without tests is an unreliable project
- Tests document expected behavior
- CI/CD (GitHub Actions) proves it works everywhere

Example from foampilot: the `AGENTS.md` documents how to run tests, which is rare in scientific projects.

---

## Lesson 6: Community as a Lever

**Lesson 6**: Your first community is your first audience.

- Respond to issues quickly (even to say "it's a bug, thanks")
- Write release notes for each version
- Use GitHub Discussions for general questions
- Highlight contributors in the README

---

## Conclusion: The Tool as a Communication Product

**Final message**: A scientific open-source project is a product with two faces:
1. The technical face (code, algorithms, rigor)
2. The communication face (README, examples, documentation, demo)

Most researchers excel at the first. Few invest enough in the second.

**CTA**: *"If you've built a CFD tool that deserves to be shared, publish it. But publish it with the same rigor as your code."*

---

*Article under improvement — version 1.0*
