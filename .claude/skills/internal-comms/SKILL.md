---
name: internal-comms
description: A set of resources to help me write all kinds of internal communications, using the formats that my company likes to use. Claude should use this skill whenever asked to write some sort of internal communications (status reports, leadership updates, 3P updates, company newsletters, FAQs, incident reports, project updates, LinkedIn messages, etc.).
license: Complete terms in LICENSE.txt
---

## When to use this skill
To write internal communications, use this skill for:
- 3P updates (Progress, Plans, Problems)
- Company newsletters
- FAQ responses
- Professional emails (academic/business correspondence)
- LinkedIn messages (networking, outreach, follow-ups)
- Status reports
- Leadership updates
- Project updates
- Incident reports

## How to use this skill

To write any internal communication:

1. **Identify the communication type** from the request
2. **Load the appropriate guideline file** from the `examples/` directory:
    - `examples/3p-updates.md` - For Progress/Plans/Problems team updates
    - `examples/company-newsletter.md` - For company-wide newsletters
    - `examples/faq-answers.md` - For answering frequently asked questions
    - `examples/professional-emails.md` - For academic/business email correspondence
    - `examples/linkedin-messages.md` - For LinkedIn networking, outreach, and follow-up messages
    - `examples/general-comms.md` - For anything else that doesn't explicitly match one of the above
3. **Follow the specific instructions** in that file for formatting, tone, and content gathering
4. **Always save the final output as a Markdown document** in the vault for easy copying

If the communication type doesn't match any existing guideline, ask for clarification or more context about the desired format.

## Output Format

**Always create a Markdown file** in a `Temporary/` directory at the project root for every communication written. Save it with:
- Proper frontmatter (date_created, date_modified, tags)
- `fileClass: Temporary`
- Clear heading with the communication type
- The full content formatted for easy copying

## Keywords
3P updates, company newsletter, company comms, weekly update, faqs, common questions, updates, internal comms, professional emails, email, academic email, LinkedIn, LinkedIn message, networking, outreach, connection request, DM
