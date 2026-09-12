# About foampilot

## Why create foampilot?

The idea behind foampilot arose from the **high learning curve of OpenFOAM**.  
Its structure, organized around many folders and dictionary files, can be complex to understand, verify, and maintain—especially in a fast-paced engineering office where quick iterations are required.

I realized that a **Python object-oriented wrapper** could help by:  
- centralizing information in readable Python files,  
- explicitly reflecting the structure of an OpenFOAM case,  
- reducing errors associated with manual dictionary editing.

Using Python tools, it becomes possible to **orchestrate the entire CFD workflow**.  
foampilot aims to provide an **open-source CFD platform**, covering the full simulation cycle:
- geometry creation,
- mesh generation,
- solver configuration and execution,
- post-processing,
- report generation.

---

## Who am I?

My name is **Steven Daix**, and I have been working in CFD for over **20 years**.

I have experience across different industries:
- nuclear,
- automotive,
- oil & gas,
- building and construction,

in companies of various sizes, from startups to large corporations, including engineering consultancies.

I am primarily a user of **Fluent (ANSYS)** and **STAR-CCM+**.  
My experience spans diverse projects, including:
- thermal draft studies in buildings,
- temperature optimization in museums,
- mixing in stirred glass baths,
- aerothermal studies under a tractor hood,
- industrial process optimization, such as toilet paper drying (fun fact!).

The difficulty of using OpenFOAM in an engineering office—for quick checks and urgent requests—has long prevented me from relying on free tools in production.

This motivated me to develop **foampilot** in my free time, based on my CFD experience and the way I wanted to work with OpenFOAM.

---

## Role of Artificial Intelligence

The development of foampilot has been **assisted by several AI tools**, including:
- ChatGPT,
- Gemini,
- Mistral,
- DeepSeek,
- Manus.

These AI tools were used as **support tools** for:
- code structuring,
- concept clarification,
- documentation improvement,
- phrasing and pedagogical clarity.

All technical choices, architecture, CFD concepts, and project vision remain **driven by my professional CFD expertise**.  
The AIs are **productivity accelerators and advisors**, not substitutes for engineering knowledge.

---

## Project Goal

foampilot aims to provide a **clear, reproducible, and automated Python interface** for OpenFOAM, enabling:
- reliable CFD studies,
- easier auditing and verification,
- increased accessibility of OpenFOAM in industrial environments,
- all while staying faithful to core CFD and OpenFOAM concepts.

> foampilot is, above all, an engineer’s project, designed by a user for users.
