# PKM SQLite Schema Reference

## Schema Rules

- Each fileClass → one main table with `_file_path TEXT PRIMARY KEY`
- `Multi` and `MultiLink` fields → junction table `{Table}_{field}` with columns `(file_path, value)`
- `Select` fields have CHECK constraints for allowed values
- `Boolean` fields stored as INTEGER (0/1)
- `Date` fields stored as TEXT in YYYY-MM-DD format
- `Link` fields store file paths as TEXT with REFERENCES

## Tables

### AgentInstruction
Folder: Agent Instructions
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| status | Select: Draft, Ready, In Progress, Completed, Archived |
| scope | Select: VS Code Extension, Script, Obsidian Plugin, CLI Tool, Full Stack, Other |
| priority | Select: High, Medium, Low |
| summary | Input (TEXT) |
| date_created | Date |
| date_modified | Date |
Junction: `AgentInstruction_tags`, `AgentInstruction_related`

### CareerDocument
Folder: CareerDocuments
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| type | Select: CV, Resume, Cover Letter, Statement of Purpose |
| format | Select: Markdown, LaTeX |
| target_role | Select: General, ML Engineer, ML Scientist, Data Analyst, Data Scientist, Robotics Engineer, Software Engineer, Research Scientist |
| target_organization | Input (TEXT) |
| version | Input (TEXT) |
| status | Select: Draft, Review, Final, Submitted, Active, Archived |
| deadline | Date |
| date_created | Date |
| date_modified | Date |
Junction: `CareerDocument_tags`

### Comparison
Folder: Knowledge
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| domain | Select: ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| verdict | Input (TEXT) |
| date_created | Date |
| date_modified | Date |
Junction: `Comparison_tags`, `Comparison_concepts_compared`, `Comparison_related_concepts`

### Concept
Folder: Knowledge
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| note_type | Select: Introduction, Deep Dive |
| domain | Select: ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| year_introduced | Number (REAL) |
| sophistication | Select: Foundational, Intermediate, Advanced, State-of-the-Art |
| date_created | Date |
| date_modified | Date |
Junction: `Concept_tags`, `Concept_antonyms`, `Concept_synonyms`, `Concept_hypernyms`, `Concept_hyponyms`, `Concept_counterparts`, `Concept_related_concepts`

### Contact
Folder: Contacts
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| name | Input (TEXT) |
| type | Select: Faculty Member, PhD, Student, OSU Staff, Consultant, Industry |
| email | Input (TEXT) |
| institute | Select: Ohio State University, University at Buffalo, Other |
| department | Select: CSE, ECE, MAE, CEG, MSE, Medicine, FABE, Psychology, Linguistics, Astronomy, COG, Law, Other |
| expertise | Input (TEXT) |
| status | Select: Need follow-up, Active, Inactive, Connected |
| office | Input (TEXT) |
| school_url | Input (TEXT) |
| website | Input (TEXT) |
Junction: `Contact_tags`

### Ideas
Folder: Ideas
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| category | Select: Search, Product, Research, Project, Other |
| status | Select: Raw, Exploring, Validated, In Progress, Archived |
| summary | Input (TEXT) |
| date_created | Date |
| date_modified | Date |
Junction: `Ideas_tags`, `Ideas_related`

### Journal
Folder: Journals
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| date | Date |
| title | Input (TEXT) |
| paper | Boolean (0/1) |
| project | Boolean (0/1) |
| networking | Boolean (0/1) |
| socializing | Boolean (0/1) |
| piano | Boolean (0/1) |
| video_games | Boolean (0/1) |
| watching | Boolean (0/1) |
| reading | Boolean (0/1) |
| improving | Boolean (0/1) |
| cardio_exercise | Boolean (0/1) |
| english | Boolean (0/1) |
| pt_exercise | Boolean (0/1) |
| podcast | Boolean (0/1) |
| class | Boolean (0/1) |
| masturbate | Boolean (0/1) |
| overgaming | Boolean (0/1) |
| overscrolling | Boolean (0/1) |
| journal_squandering | Boolean (0/1) |
Junction: `Journal_tags`

### MeetingNote
Folder: Meetings
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| date | Date |
| type | Select: One-on-one, Group meeting, Phone call, Video call, In-person, Office hours |
| status | Select: Scheduled, Completed, Cancelled, Rescheduled |
| contact | Link → Contact |
| summary | Input (TEXT) |
| duration | Input (TEXT) |
| has_transcript | Boolean (0/1) |
| has_audio | Boolean (0/1) |
Junction: `MeetingNote_tags`, `MeetingNote_participants` (MultiLink → Contact)

### Survey
Folder: Knowledge
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| domain | Select: ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| scope | Input (TEXT) |
| num_concepts | Number (REAL) |
| date_created | Date |
| date_modified | Date |
Junction: `Survey_tags`, `Survey_related_concepts`

### Temporary
Folder: Inbox
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| type | Select: Message, Status Update, 3P Update, FAQ, Newsletter, Other |
| recipient | Input (TEXT) |
| subject | Input (TEXT) |
| status | Select: Draft, Ready, Sent, Archived |
| date_created | Date |
| date_modified | Date |
Junction: `Temporary_tags`

### Tutorial
Folder: Knowledge
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| domain | Select: ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| difficulty | Select: Beginner, Intermediate, Advanced |
| estimated_time | Input (TEXT) |
| date_created | Date |
| date_modified | Date |
Junction: `Tutorial_tags`, `Tutorial_prerequisites`, `Tutorial_related_concepts`

### ProjectResumeSession
Folder: Projects
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| title | Input (TEXT) |
| institution | Select: Ohio State University, University at Buffalo, Other |
| start_date | Input (TEXT) |
| end_date | Input (TEXT) |
| project_type | Select: Class Project, Graduation Project, Hackathon, Research, Personal, Other |
| github_link | Input (TEXT) |
| status | Select: Active, Completed, Paused, Archived |
Junction: `ProjectResumeSession_tags`, `ProjectResumeSession_advisors` (MultiLink → Contact), `ProjectResumeSession_description`, `ProjectResumeSession_skills`, `ProjectResumeSession_other_links`

## Meta Tables

- `_file_metadata` (file_path, file_class, last_modified, last_indexed)
- `_validation_errors` (id, file_path, field_name, invalid_value, expected_type, timestamp)
- `_unresolved_links` (id, source_file, field_name, link_target, timestamp)
