---
title: Security for AI
date: 2026-02-08
---

What are the ways that an agent can be hijacked?

### Attack Paths

*One-Hop Attacks (Immediate Execution)*

 - **Attacks where malicious input directly influences the agent's behavior within a single interaction.** 
 - **Prompt Injection:** The attacker embeds malicious instructions directly in user input to override or manipulate the agent's behavior.
 - **Context Injection:** The attacker places malicious content in external data sources (e.g., web pages, documents, APIs). When the agent retrieves and uses this context, it unknowingly executes the injected instructions.

*Two-Hop Attacks (Delayed Execution)*

**Attacks that require an intermediate step before causing harm.**

 - **Memory Poisoning:** The attacker injects malicious content that the agent stores in memory. At a later time, the agent retrieves this poisoned memory and acts on it, triggering the attack.

### Attack Outcomes

**These attacks can lead to several classes of impact:**

 - **Tool Misuse:** The agent is manipulated into calling tools in unintended or harmful ways (e.g., sending unauthorized requests, executing destructive actions).
 - **Data Exfiltration:** Sensitive information (system prompts, user data, API keys, etc.) is extracted and exposed.
 - **Model Extraction:** The attacker attempts to replicate or infer the model's behavior, system prompt, or underlying capabilities.