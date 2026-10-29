# PolicyLens — Comprehensive Data-Cleaning Status Audit

This document is the verified data-cleaning audit across all dataset files in the PolicyLens workspace.
A total of **97 dataset files** were inspected recursively.

## Executive Summary of Dataset Holdings

| Category | File Count | Verified Cleaning Script | Primary Suitability |
|---|---|---|---|
| Primary Consultation Datasets (Raw & Cleaned) | 6 | `scripts/policylens_government_consultation_audit.py` | Unsupervised clustering & semantic discovery |
| External Corpora (Raw & Processed/Canonical) | 29 | `scripts/policylens_dataset_audit.py` | Auxiliary sentiment / emotion pretraining |
| Human Annotation Pilot Datasets | 3 | `scripts/create_annotation_pilot.py` | Supervised stance pilot / IAA evaluation |
| Model Artifacts (Embeddings & Clusters) | 3 | `scripts/concern_clustering.py` | Topic discovery & evidence retrieval |
| Interim Diagnostic & Deduplication Tables | 37 | Automated audits | Quality control & traceability |
| Academic Reference Benchmark Tables | 19 | HinEmo-9 authors | Reference baselines |
| **Total** | **97** | | |

---

## 1. Primary Government Consultation Datasets (Raw & Cleaned) (6 files)

### 1. `ap2016-18_individual_comments.csv`
- **Exact File Path**: `ap2016-18_individual_comments.csv`
- **Architectural Role**: `raw_primary`
- **File Size**: 68,438 bytes
- **Row Count**: 160 | **Columns**: 19
- **Column Names**: `Row
Rangée`, `Event`, `Événement`, `Source
`, `Source `, `Source Language`, `Langue de départ`, `Sector` ... (+11 more)
- **Missing Values**: 752 total nulls (`Other comments
Autres commentaires`: 128, `Dislike?
Ce que vous avez moins aimé ?`: 100, `Improvements? 
Améliorations`: 76, `Would you recommend?
Recommanderiez-vous `: 53, `Preferred way(s) to participate in Open Government
Façons préférées de participer au Gouvernement ouvert `: 49)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 13
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/individual_comments_2016_18_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_government_consultation_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Task Suitability**: **currently_unsuitable (raw uncleaned; use data/processed/*_source_clean.csv instead)**
  - *Rationale*: Contains multiple header rows, raw whitespace, and unstandardized blanks. Use corresponding source_clean version.

### 2. `ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv`
- **Exact File Path**: `ConsultationData2017Engagement_Donnéesdesconsultations2017engagements.csv`
- **Architectural Role**: `raw_primary`
- **File Size**: 2,982,201 bytes
- **Row Count**: 5,169 | **Columns**: 63
- **Column Names**: `ID`, `Row/Ligne`, `Event 
Événement
`, `Événement
Event (FR)`, `Source`, `Source (FR)`, `Context/Prompt
Contexte/Message-guide`, `Contexte/Message-guide
Context/Prompt Traduction si disponible` ... (+55 more)
- **Missing Values**: 122,501 total nulls (`Subject translation if available
Traduction objet si disponible`: 5,041, `A7
R2`: 4,836, `A6
R12`: 4,829, `Comment, Question or Idea translation if available
Traduction commentaire, question ou idée si disponible`: 4,642, `Subject
Objet`: 4,620)
- **Blank Text Cells (Whitespace-only)**: 1
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `Context/Prompt
Contexte/Message-guide`: 95 groups, `Contexte/Message-guide
Context/Prompt Traduction si disponible`: 91 groups, `Comment and Subject Translation Required?`: 2 groups, `Comment, Question or Idea (In source language)
Commentaire, question ou idée (dans la langue de départ)`: 24 groups, `Comment, Question or Idea translation if available
Traduction commentaire, question ou idée si disponible`: 1 groups
- **Corresponding Cleaned Version**: `data/processed/consultation_compilation_2017_18_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_government_consultation_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Task Suitability**: **currently_unsuitable (raw uncleaned; use data/processed/*_source_clean.csv instead)**
  - *Rationale*: Contains multiple header rows, raw whitespace, and unstandardized blanks. Use corresponding source_clean version.

### 3. `consultation_compilation_2017_18_source_clean.csv`
- **Exact File Path**: `data/processed/consultation_compilation_2017_18_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 3,598,665 bytes
- **Row Count**: 5,169 | **Columns**: 68
- **Column Names**: `source_dataset`, `source_file`, `source_row_number`, `duplicate_text_group_id`, `exclusion_reason`, `id`, `row_ligne`, `event_v_nement` ... (+60 more)
- **Missing Values**: 132,786 total nulls (`exclusion_reason`: 5,169, `duplicate_text_group_id`: 5,115, `subject_translation_if_available_traduction_objet_si_disponible`: 5,041, `a7_r2`: 4,836, `a6_r12`: 4,829)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `duplicate_text_group_id`: 24 groups, `context_prompt_contexte_message_guide`: 95 groups, `contexte_message_guide_context_prompt_traduction_si_disponible`: 91 groups, `comment_and_subject_translation_required`: 2 groups, `comment_question_or_idea_in_source_language_commentaire_question_ou_id_e_dans_la_langue_de_d_part`: 25 groups, `comment_question_or_idea_translation_if_available_traduction_commentaire_question_ou_id_e_si_disponible`: 1 groups
- **Corresponding Cleaned Version**: `data/processed/consultation_compilation_2017_18_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'English': 4851, 'French': 315, 'Arabic': 2, 'Japanese': 1}
- **Task Suitability**: **unsupervised_clustering & semantic_retrieval**
  - *Rationale*: Source-cleaned Canadian federal open government consultation responses with normalized whitespace, standardized nulls, and preserved metadata; suitable for unsupervised clustering and semantic search.

### 4. `get_involved_questionnaire_2017_18_source_clean.csv`
- **Exact File Path**: `data/processed/get_involved_questionnaire_2017_18_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 4,015,948 bytes
- **Row Count**: 8,212 | **Columns**: 103
- **Column Names**: `source_dataset`, `source_file`, `source_row_number`, `duplicate_text_group_id`, `exclusion_reason`, `unique_identifier`, `language`, `start_date` ... (+95 more)
- **Missing Values**: 546,131 total nulls (`exclusion_reason`: 8,212, `q2_what_are_your_three_most_important_government_transparency_topics`: 8,212, `q4_what_are_your_three_most_important_government_accountability_needs`: 8,212, `q6_what_are_the_three_most_important_ways_to_participate_and_interact_with_government`: 8,212, `q7_the_federal_government_works_to_improve_the_16_strategic_outcomes_listed_below_what_are_the_three_most_important_outcomes_to_be_included_in_canada_s_4th_plan_on_open_government`: 8,212)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `duplicate_text_group_id`: 40 groups, `q8_did_we_miss_anything_if_the_previous_questions_did_not_speak_to_you_please_tell_us_about_the_important_aspects_of_open_government_that_drive_your_interest_if_you_have_more_than_one_additional_item_please_list_them_in_priority`: 18 groups
- **Corresponding Cleaned Version**: `data/processed/get_involved_questionnaire_2017_18_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'English /\nAnglais': 6803, 'French / français': 1409}
- **Task Suitability**: **unsupervised_clustering & semantic_retrieval**
  - *Rationale*: Source-cleaned Canadian federal open government consultation responses with normalized whitespace, standardized nulls, and preserved metadata; suitable for unsupervised clustering and semantic search.

### 5. `individual_comments_2016_18_source_clean.csv`
- **Exact File Path**: `data/processed/individual_comments_2016_18_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 79,471 bytes
- **Row Count**: 160 | **Columns**: 24
- **Column Names**: `source_dataset`, `source_file`, `source_row_number`, `duplicate_text_group_id`, `exclusion_reason`, `row_rang_e`, `event`, `v_nement` ... (+16 more)
- **Missing Values**: 1,059 total nulls (`exclusion_reason`: 160, `duplicate_text_group_id`: 146, `other_comments_autres_commentaires`: 128, `dislike_ce_que_vous_avez_moins_aim`: 101, `improvements_am_liorations`: 76)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `duplicate_text_group_id`: 1 groups
- **Corresponding Cleaned Version**: `data/processed/individual_comments_2016_18_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'English': 139, 'French': 6, 'S': 1}
- **Task Suitability**: **unsupervised_clustering & semantic_retrieval**
  - *Rationale*: Source-cleaned Canadian federal open government consultation responses with normalized whitespace, standardized nulls, and preserved metadata; suitable for unsupervised clustering and semantic search.

### 6. `Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv`
- **Exact File Path**: `Open Government Consultation Data 2017-18 Get Involved Questionnaire.csv`
- **Architectural Role**: `raw_primary`
- **File Size**: 3,048,593 bytes
- **Row Count**: 8,213 | **Columns**: 98
- **Column Names**: `
Identifiant unique`, `
La langue`, `
Date de début`, `Date de fin`, `Q1. Est-ce que la transparence est un élément important pour vous?`, `Q2. Quels sont les trois domaines de la transparence du gouvernement les plus importants pour vous?`, `Q2R. Être en mesure de trouver la personne à qui je dois parler dans le gouvernement.`, `Q2R. S’informer sur les politiques et les programmes du gouvernement.` ... (+90 more)
- **Missing Values**: 533,911 total nulls (`Q2. Quels sont les trois domaines de la transparence du gouvernement les plus importants pour vous?`: 8,212, `Q4. Quels sont vos principaux besoins en matière de la responsabilisation du gouvernement?`: 8,212, `Q6. Quels sont les principaux moyens de participer et d’interagir avec le gouvernement?  `: 8,212, `Q7. Le gouvernement fédéral travaille à l’amélioration des 16 résultats stratégiques indiqués ci-dessous. Quels sont les trois plus importants résultats qui devraient être inclus dans le 4e Plan du Canada pour un gouvernement ouvert?`: 8,212, `Q9. Laquelle des déclarations suivantes décrit le mieux votre intérêt concernant le gouvernement ouvert?`: 8,212)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `Q6R. Être en mesure d’interagir avec d’autres Canadiens sur des questions faisant l’objet de préoccupations communes.`: 1 groups, `Q6R.  Être en mesure de collaborer directement avec le gouvernement sur des questions importantes pour moi.`: 1 groups, `Q8. Avons-nous omis quelque chose?  Si les questions précédentes ne vous ont pas interpellé, veuillez nous indiquer quels aspects essentiels du gouvernement ouvert vous intéressent :Si vous avez plus d’un point supplémentaire, veuillez les indiquer dans l’ordre de leur priorité.`: 18 groups, `Q9R. Je fournis des commentaires au gouvernement uniquement lorsqu’on m’interroge sur une question qui me tient à cœur.`: 1 groups, `Q9R. Je compile les données et les renseignements autour d’une question particulière.`: 1 groups, `Q11R.  J’aimerais vous offrir des commentaires sur les ébauches de documents.`: 3 groups, `Q12R. Des questionnaires semblables à celui-ci`: 1 groups, `Q20. Je réponds à ce questionnaire au nom d’une organisation :`: 3 groups
- **Corresponding Cleaned Version**: `data/processed/get_involved_questionnaire_2017_18_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_government_consultation_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Task Suitability**: **currently_unsuitable (raw uncleaned; use data/processed/*_source_clean.csv instead)**
  - *Rationale*: Contains multiple header rows, raw whitespace, and unstandardized blanks. Use corresponding source_clean version.

## 2. External Corpora & Benchmarks (Raw & Processed/Canonical) (29 files)

### 1. `all_detected_datasets_canonical.csv`
- **Exact File Path**: `data/processed/all_detected_datasets_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 428,210,282 bytes
- **Row Count**: 1,652,591 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/all_detected_datasets_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Task Suitability**: **unsuitable**
  - *Rationale*: 

### 2. `hinemo9_dev_canonical.csv`
- **Exact File Path**: `data/processed/hinemo9_dev_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 264,830 bytes
- **Row Count**: 1,009 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 5,045 total nulls (`consultation_id`: 1,009, `question_id`: 1,009, `question_text`: 1,009, `duplicate_group_id`: 1,009, `exclusion_reason`: 1,009)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_dev_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'Hinglish': 393, 'Hindi': 343, 'English': 273}
- **Label Distribution**: {'Surprise': 200, 'Fear': 160, 'Love': 126, 'Joy': 115, 'Anger': 94}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 3. `hinemo9_dev_source_clean.csv`
- **Exact File Path**: `data/processed/hinemo9_dev_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 285,152 bytes
- **Row Count**: 1,009 | **Columns**: 11
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `n_annotators`, `gold_source` ... (+3 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_dev_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'Hinglish': 393, 'Hindi': 343, 'English': 273}
- **Label Distribution**: {'Surprise': 200, 'Fear': 160, 'Love': 126, 'Joy': 115, 'Anger': 94}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 4. `hinemo9_full_canonical.csv`
- **Exact File Path**: `data/processed/hinemo9_full_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 7,964,935 bytes
- **Row Count**: 30,436 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 152,174 total nulls (`consultation_id`: 30,436, `question_id`: 30,436, `question_text`: 30,436, `exclusion_reason`: 30,436, `duplicate_group_id`: 30,430)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `cleaned_text`: 1 groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_full_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'English': 11004, 'Hinglish': 10393, 'Hindi': 9039}
- **Label Distribution**: {'Neutral': 4500, 'Love': 3600, 'Devotion': 3600, 'Joy': 3600, 'Sadness': 3600}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 5. `hinemo9_full_source_clean.csv`
- **Exact File Path**: `data/processed/hinemo9_full_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 6,815,638 bytes
- **Row Count**: 30,436 | **Columns**: 9
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `code_mix_index`, `n_tokens`, `label_source`, `cleaned_text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `cleaned_text`: 1 groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_full_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'English': 11004, 'Hinglish': 10393, 'Hindi': 9039}
- **Label Distribution**: {'Neutral': 4500, 'Love': 3600, 'Devotion': 3600, 'Joy': 3600, 'Sadness': 3600}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 6. `hinemo9_gold_canonical.csv`
- **Exact File Path**: `data/processed/hinemo9_gold_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 898,013 bytes
- **Row Count**: 3,363 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 16,815 total nulls (`consultation_id`: 3,363, `question_id`: 3,363, `question_text`: 3,363, `duplicate_group_id`: 3,363, `exclusion_reason`: 3,363)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_gold_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'Hinglish': 1308, 'Hindi': 1149, 'English': 906}
- **Label Distribution**: {'Surprise': 666, 'Fear': 532, 'Love': 418, 'Joy': 384, 'Anger': 317}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 7. `hinemo9_gold_source_clean.csv`
- **Exact File Path**: `data/processed/hinemo9_gold_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 956,723 bytes
- **Row Count**: 3,363 | **Columns**: 11
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `gold_source`, `n_annotators` ... (+3 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_gold_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'Hinglish': 1308, 'Hindi': 1149, 'English': 906}
- **Label Distribution**: {'Surprise': 666, 'Fear': 532, 'Love': 418, 'Joy': 384, 'Anger': 317}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 8. `hinemo9_test_canonical.csv`
- **Exact File Path**: `data/processed/hinemo9_test_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 272,459 bytes
- **Row Count**: 1,010 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 5,050 total nulls (`consultation_id`: 1,010, `question_id`: 1,010, `question_text`: 1,010, `duplicate_group_id`: 1,010, `exclusion_reason`: 1,010)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_test_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'Hinglish': 393, 'Hindi': 347, 'English': 270}
- **Label Distribution**: {'Surprise': 200, 'Fear': 159, 'Love': 125, 'Joy': 115, 'Anger': 96}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 9. `hinemo9_test_source_clean.csv`
- **Exact File Path**: `data/processed/hinemo9_test_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 290,947 bytes
- **Row Count**: 1,010 | **Columns**: 11
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `n_annotators`, `gold_source` ... (+3 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_test_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'Hinglish': 393, 'Hindi': 347, 'English': 270}
- **Label Distribution**: {'Surprise': 200, 'Fear': 159, 'Love': 125, 'Joy': 115, 'Anger': 96}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 10. `hinemo9_train_canonical.csv`
- **Exact File Path**: `data/processed/hinemo9_train_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 7,477,534 bytes
- **Row Count**: 28,417 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 142,079 total nulls (`consultation_id`: 28,417, `question_id`: 28,417, `question_text`: 28,417, `exclusion_reason`: 28,417, `duplicate_group_id`: 28,411)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `cleaned_text`: 1 groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_train_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'English': 10461, 'Hinglish': 9607, 'Hindi': 8349}
- **Label Distribution**: {'Neutral': 4218, 'Love': 3471, 'Joy': 3460, 'Nostalgia': 3420, 'Sadness': 3408}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 11. `hinemo9_train_source_clean.csv`
- **Exact File Path**: `data/processed/hinemo9_train_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 6,426,987 bytes
- **Row Count**: 28,417 | **Columns**: 11
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `n_annotators`, `gold_source` ... (+3 more)
- **Missing Values**: 81,219 total nulls (`gold_source`: 27,073, `rater_labels`: 27,073, `llm_prediction`: 27,073)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `cleaned_text`: 1 groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_train_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Language Distribution**: {'English': 10461, 'Hinglish': 9607, 'Hindi': 8349}
- **Label Distribution**: {'Neutral': 4218, 'Love': 3471, 'Joy': 3460, 'Nostalgia': 3420, 'Sadness': 3408}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 12. `policylens_sentiment_canonical.csv`
- **Exact File Path**: `data/processed/policylens_sentiment_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 680,131 bytes
- **Row Count**: 2,500 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 17,496 total nulls (`source_record_id`: 2,500, `consultation_id`: 2,500, `question_id`: 2,500, `question_text`: 2,500, `language`: 2,500)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `original_text`: 2 groups, `cleaned_text`: 2 groups
- **Corresponding Cleaned Version**: `data/processed/policylens_sentiment_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'positive': 955, 'negative': 848, 'neutral': 697}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 13. `policylens_sentiment_source_clean.csv`
- **Exact File Path**: `data/processed/policylens_sentiment_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 468,545 bytes
- **Row Count**: 2,500 | **Columns**: 3
- **Column Names**: `sentence`, `sentiment`, `cleaned_text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `sentence`: 2 groups, `cleaned_text`: 2 groups
- **Corresponding Cleaned Version**: `data/processed/policylens_sentiment_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'positive': 955, 'negative': 848, 'neutral': 697}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 14. `sentiment140_test_canonical.csv`
- **Exact File Path**: `data/processed/sentiment140_test_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 126,581 bytes
- **Row Count**: 498 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 2,988 total nulls (`consultation_id`: 498, `question_id`: 498, `question_text`: 498, `language`: 498, `duplicate_group_id`: 498)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentiment140_test_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'positive': 182, 'negative': 177, 'neutral': 139}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: English Twitter polarity dataset (1.6M tweets); domain mismatch with government policy consultations; useful solely for polarity baseline pretraining.

### 15. `sentiment140_test_source_clean.csv`
- **Exact File Path**: `data/processed/sentiment140_test_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 109,475 bytes
- **Row Count**: 498 | **Columns**: 7
- **Column Names**: `target`, `ids`, `date`, `flag`, `user`, `text`, `cleaned_text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentiment140_test_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'4': 182, '0': 177, '2': 139}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: English Twitter polarity dataset (1.6M tweets); domain mismatch with government policy consultations; useful solely for polarity baseline pretraining.

### 16. `sentiment140_train_canonical.csv`
- **Exact File Path**: `data/processed/sentiment140_train_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 404,165,409 bytes
- **Row Count**: 1,600,000 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentiment140_train_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'negative': 800000, 'positive': 800000}
- **Task Suitability**: **unsuitable**
  - *Rationale*: 

### 17. `sentiment140_train_source_clean.csv`
- **Exact File Path**: `data/processed/sentiment140_train_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 339,602,535 bytes
- **Row Count**: 1,600,000 | **Columns**: 7
- **Column Names**: `target`, `ids`, `date`, `flag`, `user`, `text`, `cleaned_text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `text`: 23553 groups
- **Corresponding Cleaned Version**: `data/processed/sentiment140_train_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'0': 800000, '4': 800000}
- **Task Suitability**: **unsuitable**
  - *Rationale*: 

### 18. `sentimix_test_canonical.csv`
- **Exact File Path**: `data/processed/sentimix_test_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 1,069,989 bytes
- **Row Count**: 3,000 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 18,000 total nulls (`consultation_id`: 3,000, `question_id`: 3,000, `question_text`: 3,000, `language`: 3,000, `duplicate_group_id`: 3,000)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentimix_test_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'neutral': 1100, 'positive': 1000, 'negative': 900}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 19. `sentimix_test_source_clean.csv`
- **Exact File Path**: `data/processed/sentimix_test_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 1,151,842 bytes
- **Row Count**: 3,000 | **Columns**: 6
- **Column Names**: `uid`, `text`, `token_count`, `token_language_tags`, `sentiment`, `cleaned_text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentimix_test_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'neutral': 1100, 'positive': 1000, 'negative': 900}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 20. `sentimix_train_canonical.csv`
- **Exact File Path**: `data/processed/sentimix_train_canonical.csv`
- **Architectural Role**: `processed_canonical`
- **File Size**: 5,292,066 bytes
- **Row Count**: 15,131 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 90,786 total nulls (`consultation_id`: 15,131, `question_id`: 15,131, `question_text`: 15,131, `language`: 15,131, `duplicate_group_id`: 15,131)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentimix_train_canonical.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'neutral': 5638, 'positive': 5034, 'negative': 4459}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 21. `sentimix_train_source_clean.csv`
- **Exact File Path**: `data/processed/sentimix_train_source_clean.csv`
- **Architectural Role**: `processed_cleaned`
- **File Size**: 5,667,675 bytes
- **Row Count**: 15,131 | **Columns**: 6
- **Column Names**: `uid`, `text`, `token_count`, `token_language_tags`, `sentiment`, `cleaned_text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentimix_train_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: Generated by scripts/policylens_dataset_audit.py or scripts/policylens_government_consultation_audit.py
- **Label Distribution**: {'neutral': 5638, 'positive': 5034, 'negative': 4459}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 22. `dev.jsonl`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/dev.jsonl`
- **Architectural Role**: `other`
- **File Size**: 356,183 bytes
- **Row Count**: 1,009 | **Columns**: 10
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `n_annotators`, `gold_source`, `rater_labels`, `llm_prediction`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_dev_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Language Distribution**: {'Hinglish': 393, 'Hindi': 343, 'English': 273}
- **Label Distribution**: {'Surprise': 200, 'Fear': 160, 'Love': 126, 'Joy': 115, 'Anger': 94}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 23. `full.jsonl`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/full.jsonl`
- **Architectural Role**: `other`
- **File Size**: 7,842,692 bytes
- **Row Count**: 30,436 | **Columns**: 8
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `code_mix_index`, `n_tokens`, `label_source`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_full_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Language Distribution**: {'English': 11004, 'Hinglish': 10393, 'Hindi': 9039}
- **Label Distribution**: {'Neutral': 4500, 'Love': 3600, 'Devotion': 3600, 'Joy': 3600, 'Sadness': 3600}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 24. `gold.jsonl`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/gold.jsonl`
- **Architectural Role**: `other`
- **File Size**: 1,190,480 bytes
- **Row Count**: 3,363 | **Columns**: 10
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `gold_source`, `n_annotators`, `rater_labels`, `llm_prediction`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_gold_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Language Distribution**: {'Hinglish': 1308, 'Hindi': 1149, 'English': 906}
- **Label Distribution**: {'Surprise': 666, 'Fear': 532, 'Love': 418, 'Joy': 384, 'Anger': 317}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 25. `test.jsonl`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/test.jsonl`
- **Architectural Role**: `other`
- **File Size**: 359,353 bytes
- **Row Count**: 1,010 | **Columns**: 10
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `n_annotators`, `gold_source`, `rater_labels`, `llm_prediction`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_test_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Language Distribution**: {'Hinglish': 393, 'Hindi': 347, 'English': 270}
- **Label Distribution**: {'Surprise': 200, 'Fear': 159, 'Love': 125, 'Joy': 115, 'Anger': 96}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 26. `train.jsonl`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/data/train.jsonl`
- **Architectural Role**: `other`
- **File Size**: 6,958,190 bytes
- **Row Count**: 28,417 | **Columns**: 10
- **Column Names**: `id`, `text`, `label`, `language_label`, `genre`, `label_source`, `n_annotators`, `gold_source`, `rater_labels`, `llm_prediction`
- **Missing Values**: 0 total nulls (`gold_source`: 27,073, `rater_labels`: 27,073, `llm_prediction`: 27,073)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/hinemo9_train_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Language Distribution**: {'English': 10461, 'Hinglish': 9607, 'Hindi': 8349}
- **Label Distribution**: {'Neutral': 4218, 'Love': 3471, 'Joy': 3460, 'Nostalgia': 3420, 'Sadness': 3408}
- **Task Suitability**: **auxiliary_emotion_experiments**
  - *Rationale*: Multilingual Hindi/Hinglish YouTube comment emotion classification; not consultation stance.

### 27. `PolicyLens_Hindi_English_Hinglish_Sentiment.csv`
- **Exact File Path**: `PolicyLens_Hindi_English_Hinglish_Sentiment.csv`
- **Architectural Role**: `raw_primary`
- **File Size**: 247,681 bytes
- **Row Count**: 2,500 | **Columns**: 2
- **Column Names**: `ï»¿sentence`, `sentiment`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `ï»¿sentence`: 2 groups
- **Corresponding Cleaned Version**: `data/processed/policylens_sentiment_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Label Distribution**: {'positive': 955, 'negative': 848, 'neutral': 697}
- **Task Suitability**: **currently_unsuitable**
  - *Rationale*: Not a model training or retrieval corpus.

### 28. `testdata.manual.2009.06.14.csv`
- **Exact File Path**: `trainingandtestdata/testdata.manual.2009.06.14.csv`
- **Architectural Role**: `other`
- **File Size**: 74,326 bytes
- **Row Count**: 497 | **Columns**: 6
- **Column Names**: `4`, `3`, `Mon May 11 03:17:40 UTC 2009`, `kindle2`, `tpryan`, `@stellargirl I loooooooovvvvvveee my Kindle2. Not that the DX is cool, but the 2 is fantastic in its own right.`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentiment140_test_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Task Suitability**: **currently_unsuitable**
  - *Rationale*: Not a model training or retrieval corpus.

### 29. `training.1600000.processed.noemoticon.csv`
- **Exact File Path**: `trainingandtestdata/training.1600000.processed.noemoticon.csv`
- **Architectural Role**: `other`
- **File Size**: 238,803,811 bytes
- **Row Count**: 1,600,000 | **Columns**: 6
- **Column Names**: `0`, `1467810369`, `Mon Apr 06 22:19:45 PDT 2009`, `NO_QUERY`, `_TheSpecialOne_`, `@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: `data/processed/sentiment140_train_source_clean.csv`
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/policylens_dataset_audit.py, notebooks/01_dataset_audit_and_cleaning.ipynb
- **Label Distribution**: {'0': 800000, '4': 800000}
- **Task Suitability**: **unsuitable**
  - *Rationale*: 

## 3. Human Annotation Pilot Datasets (3 files)

### 1. `annotator_1.csv`
- **Exact File Path**: `data/annotation/annotator_1.csv`
- **Architectural Role**: `annotated`
- **File Size**: 28,481 bytes
- **Row Count**: 50 | **Columns**: 18
- **Column Names**: `pilot_id`, `source_dataset`, `source_record_id`, `source_file`, `source_row_number`, `source_field`, `question_slug`, `question_wording` ... (+10 more)
- **Missing Values**: 349 total nulls (`stance`: 50, `concern`: 50, `requested_action`: 50, `evidence_span`: 50, `review_notes`: 50)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `question_slug`: 9 groups, `question_wording`: 9 groups, `question_context`: 5 groups, `context`: 5 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/create_annotation_templates.py, notebooks/03_annotation_agreement_analysis.ipynb
- **Task Suitability**: **supervised_stance_training (evaluation / pilot)**
  - *Rationale*: Contains 50 multi-annotated consultation responses with stance labels and evidence spans; suitable for inter-annotator agreement evaluation, prompt benchmarking, and few-shot calibration.

### 2. `annotator_2.csv`
- **Exact File Path**: `data/annotation/annotator_2.csv`
- **Architectural Role**: `annotated`
- **File Size**: 28,481 bytes
- **Row Count**: 50 | **Columns**: 18
- **Column Names**: `pilot_id`, `source_dataset`, `source_record_id`, `source_file`, `source_row_number`, `source_field`, `question_slug`, `question_wording` ... (+10 more)
- **Missing Values**: 349 total nulls (`stance`: 50, `concern`: 50, `requested_action`: 50, `evidence_span`: 50, `review_notes`: 50)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `question_slug`: 9 groups, `question_wording`: 9 groups, `question_context`: 5 groups, `context`: 5 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/create_annotation_templates.py, notebooks/03_annotation_agreement_analysis.ipynb
- **Task Suitability**: **supervised_stance_training (evaluation / pilot)**
  - *Rationale*: Contains 50 multi-annotated consultation responses with stance labels and evidence spans; suitable for inter-annotator agreement evaluation, prompt benchmarking, and few-shot calibration.

### 3. `pilot_annotation_50.csv`
- **Exact File Path**: `data/annotation/pilot_annotation_50.csv`
- **Architectural Role**: `annotated`
- **File Size**: 28,481 bytes
- **Row Count**: 50 | **Columns**: 18
- **Column Names**: `pilot_id`, `source_dataset`, `source_record_id`, `source_file`, `source_row_number`, `source_field`, `question_slug`, `question_wording` ... (+10 more)
- **Missing Values**: 349 total nulls (`stance`: 50, `concern`: 50, `requested_action`: 50, `evidence_span`: 50, `review_notes`: 50)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `question_slug`: 9 groups, `question_wording`: 9 groups, `question_context`: 5 groups, `context`: 5 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/create_annotation_pilot.py, notebooks/03_annotation_agreement_analysis.ipynb
- **Task Suitability**: **supervised_stance_training (evaluation / pilot)**
  - *Rationale*: Contains 50 multi-annotated consultation responses with stance labels and evidence spans; suitable for inter-annotator agreement evaluation, prompt benchmarking, and few-shot calibration.

## 4. Semantic Embeddings & Concern Clustering Pipeline Artifacts (3 files)

### 1. `cluster_labels.csv`
- **Exact File Path**: `data/clustering/cluster_labels.csv`
- **Architectural Role**: `model_generated_artifact`
- **File Size**: 221,764 bytes
- **Row Count**: 450 | **Columns**: 17
- **Column Names**: `corpus_id`, `source_dataset`, `source_record_id`, `source_file`, `source_row_number`, `source_field`, `question_slug`, `question_wording` ... (+9 more)
- **Missing Values**: 452 total nulls (`duplicate_text_group_id`: 450, `stakeholder_group`: 2)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `question_slug`: 9 groups, `question_wording`: 9 groups, `question_context`: 56 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/concern_clustering.py, notebooks/05_concern_clustering.ipynb
- **Language Distribution**: {'English': 417, 'French': 33}
- **Task Suitability**: **unsupervised_clustering (active artifact)**
  - *Rationale*: 450 deduplicated, length-filtered consultation comments with MiniLM embeddings, UMAP coordinates, and HDBSCAN topic clusters.

### 2. `cluster_topics.csv`
- **Exact File Path**: `data/clustering/cluster_topics.csv`
- **Architectural Role**: `model_generated_artifact`
- **File Size**: 605 bytes
- **Row Count**: 3 | **Columns**: 5
- **Column Names**: `cluster_id`, `size`, `top_keywords`, `language_distribution`, `source_distribution`
- **Missing Values**: 1 total nulls (`top_keywords`: 1)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/concern_clustering.py, notebooks/05_concern_clustering.ipynb
- **Task Suitability**: **currently_unsuitable**
  - *Rationale*: Not a model training or retrieval corpus.

### 3. `corpus_metadata.csv`
- **Exact File Path**: `data/embeddings/corpus_metadata.csv`
- **Architectural Role**: `model_generated_artifact`
- **File Size**: 211,343 bytes
- **Row Count**: 450 | **Columns**: 13
- **Column Names**: `corpus_id`, `source_dataset`, `source_record_id`, `source_file`, `source_row_number`, `source_field`, `question_slug`, `question_wording` ... (+5 more)
- **Missing Values**: 452 total nulls (`duplicate_text_group_id`: 450, `stakeholder_group`: 2)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `question_slug`: 9 groups, `question_wording`: 9 groups, `question_context`: 56 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [VERIFIED]
  - **Verification Source**: scripts/semantic_similarity.py, notebooks/04_semantic_similarity.ipynb
- **Language Distribution**: {'English': 417, 'French': 33}
- **Task Suitability**: **unsupervised_clustering (active artifact)**
  - *Rationale*: 450 deduplicated, length-filtered consultation comments with MiniLM embeddings, UMAP coordinates, and HDBSCAN topic clusters.

## 5. Interim Diagnostic, Mapping & Deduplication Reviews (37 files)

### 1. `consultation_compilation_2017_18_column_mapping.csv`
- **Exact File Path**: `data/interim/consultation_compilation_2017_18_column_mapping.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 4,818 bytes
- **Row Count**: 63 | **Columns**: 3
- **Column Names**: `dataset_name`, `original_column_name`, `normalized_column_name`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 2. `consultation_compilation_2017_18_header_rows.csv`
- **Exact File Path**: `data/interim/consultation_compilation_2017_18_header_rows.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 1,422 bytes
- **Row Count**: 0 | **Columns**: 63
- **Column Names**: `ID`, `Row/Ligne`, `Event 
Événement
`, `Événement
Event (FR)`, `Source`, `Source (FR)`, `Context/Prompt
Contexte/Message-guide`, `Contexte/Message-guide
Context/Prompt Traduction si disponible` ... (+55 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 3. `consultation_compilation_2017_18_text_entries.csv`
- **Exact File Path**: `data/interim/consultation_compilation_2017_18_text_entries.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 2,454,385 bytes
- **Row Count**: 5,695 | **Columns**: 6
- **Column Names**: `dataset_name`, `source_row_index`, `source_row_number`, `column_name`, `text_key`, `text_sample_redacted`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `text_key`: 27 groups, `text_sample_redacted`: 27 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 4. `consultation_compilation_2017_18_text_fields.csv`
- **Exact File Path**: `data/interim/consultation_compilation_2017_18_text_fields.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 15,013 bytes
- **Row Count**: 63 | **Columns**: 10
- **Column Names**: `column_name`, `non_empty_responses`, `missing_responses`, `avg_length`, `max_length`, `unique_values`, `field_type`, `policy_relevance_recommendation`, `representative_samples_redacted`, `dataset_name`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 5. `get_involved_questionnaire_2017_18_column_mapping.csv`
- **Exact File Path**: `data/interim/get_involved_questionnaire_2017_18_column_mapping.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 15,260 bytes
- **Row Count**: 98 | **Columns**: 3
- **Column Names**: `dataset_name`, `original_column_name`, `normalized_column_name`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 6. `get_involved_questionnaire_2017_18_header_rows.csv`
- **Exact File Path**: `data/interim/get_involved_questionnaire_2017_18_header_rows.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 13,760 bytes
- **Row Count**: 1 | **Columns**: 98
- **Column Names**: `
Identifiant unique`, `
La langue`, `
Date de début`, `Date de fin`, `Q1. Est-ce que la transparence est un élément important pour vous?`, `Q2. Quels sont les trois domaines de la transparence du gouvernement les plus importants pour vous?`, `Q2R. Être en mesure de trouver la personne à qui je dois parler dans le gouvernement.`, `Q2R. S’informer sur les politiques et les programmes du gouvernement.` ... (+90 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 7. `get_involved_questionnaire_2017_18_text_entries.csv`
- **Exact File Path**: `data/interim/get_involved_questionnaire_2017_18_text_entries.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 2,977,700 bytes
- **Row Count**: 7,964 | **Columns**: 6
- **Column Names**: `dataset_name`, `source_row_index`, `source_row_number`, `column_name`, `text_key`, `text_sample_redacted`
- **Missing Values**: 1 total nulls (`text_key`: 1)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `text_key`: 151 groups, `text_sample_redacted`: 151 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 8. `get_involved_questionnaire_2017_18_text_fields.csv`
- **Exact File Path**: `data/interim/get_involved_questionnaire_2017_18_text_fields.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 28,088 bytes
- **Row Count**: 98 | **Columns**: 10
- **Column Names**: `column_name`, `non_empty_responses`, `missing_responses`, `avg_length`, `max_length`, `unique_values`, `field_type`, `policy_relevance_recommendation`, `representative_samples_redacted`, `dataset_name`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 9. `government_duplicate_text_entries.csv`
- **Exact File Path**: `data/interim/government_duplicate_text_entries.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 99,541 bytes
- **Row Count**: 596 | **Columns**: 6
- **Column Names**: `dataset_name`, `source_row_index`, `source_row_number`, `column_name`, `text_key`, `text_sample_redacted`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `text_key`: 188 groups, `text_sample_redacted`: 188 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 10. `government_id_overlap.csv`
- **Exact File Path**: `data/interim/government_id_overlap.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 353 bytes
- **Row Count**: 1 | **Columns**: 6
- **Column Names**: `left_dataset`, `left_id_column`, `right_dataset`, `right_id_column`, `overlapping_ids`, `sample_ids`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 11. `government_near_duplicate_candidates.csv`
- **Exact File Path**: `data/interim/government_near_duplicate_candidates.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 8,603 bytes
- **Row Count**: 19 | **Columns**: 9
- **Column Names**: `left_dataset`, `left_row`, `left_column`, `right_dataset`, `right_row`, `right_column`, `similarity`, `left_sample_redacted`, `right_sample_redacted`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 12. `government_raw_file_checksums.csv`
- **Exact File Path**: `data/interim/government_raw_file_checksums.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 507 bytes
- **Row Count**: 3 | **Columns**: 3
- **Column Names**: `dataset_name`, `relative_path`, `sha256`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 13. `government_response_text_entries.csv`
- **Exact File Path**: `data/interim/government_response_text_entries.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 5,560,079 bytes
- **Row Count**: 14,135 | **Columns**: 6
- **Column Names**: `dataset_name`, `source_row_index`, `source_row_number`, `column_name`, `text_key`, `text_sample_redacted`
- **Missing Values**: 1 total nulls (`text_key`: 1)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `text_key`: 188 groups, `text_sample_redacted`: 188 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 14. `government_row_count_verification.csv`
- **Exact File Path**: `data/interim/government_row_count_verification.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 186 bytes
- **Row Count**: 3 | **Columns**: 4
- **Column Names**: `dataset_name`, `input_rows`, `source_clean_rows`, `excluded_rows`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 15. `government_schema_overlap.csv`
- **Exact File Path**: `data/interim/government_schema_overlap.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 358 bytes
- **Row Count**: 3 | **Columns**: 6
- **Column Names**: `left_dataset`, `right_dataset`, `left_columns`, `right_columns`, `common_normalized_columns`, `common_column_names`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 16. `government_text_field_assessment.csv`
- **Exact File Path**: `data/interim/government_text_field_assessment.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 50,880 bytes
- **Row Count**: 180 | **Columns**: 11
- **Column Names**: `column_name`, `non_empty_responses`, `missing_responses`, `missing_pct`, `avg_length`, `max_length`, `unique_values`, `field_type` ... (+3 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 17. `hinemo9_dev_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/hinemo9_dev_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 185 bytes
- **Row Count**: 0 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 18. `hinemo9_full_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/hinemo9_full_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 2,181 bytes
- **Row Count**: 6 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 24 total nulls (`consultation_id`: 6, `question_id`: 6, `question_text`: 6, `exclusion_reason`: 6)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `cleaned_text`: 1 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Language Distribution**: {'Hinglish': 4, 'English': 2}
- **Label Distribution**: {'Joy': 4, 'Fear': 2}
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 19. `hinemo9_gold_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/hinemo9_gold_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 185 bytes
- **Row Count**: 0 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 20. `hinemo9_test_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/hinemo9_test_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 185 bytes
- **Row Count**: 0 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 21. `hinemo9_train_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/hinemo9_train_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 2,194 bytes
- **Row Count**: 6 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 24 total nulls (`consultation_id`: 6, `question_id`: 6, `question_text`: 6, `exclusion_reason`: 6)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `cleaned_text`: 1 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Language Distribution**: {'Hinglish': 4, 'English': 2}
- **Label Distribution**: {'Joy': 4, 'Fear': 2}
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 22. `individual_comments_2016_18_column_mapping.csv`
- **Exact File Path**: `data/interim/individual_comments_2016_18_column_mapping.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 1,944 bytes
- **Row Count**: 19 | **Columns**: 3
- **Column Names**: `dataset_name`, `original_column_name`, `normalized_column_name`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 23. `individual_comments_2016_18_header_rows.csv`
- **Exact File Path**: `data/interim/individual_comments_2016_18_header_rows.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 701 bytes
- **Row Count**: 0 | **Columns**: 19
- **Column Names**: `Row
Rangée`, `Event`, `Événement`, `Source
`, `Source `, `Source Language`, `Langue de départ`, `Sector` ... (+11 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 24. `individual_comments_2016_18_text_entries.csv`
- **Exact File Path**: `data/interim/individual_comments_2016_18_text_entries.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 128,190 bytes
- **Row Count**: 476 | **Columns**: 6
- **Column Names**: `dataset_name`, `source_row_index`, `source_row_number`, `column_name`, `text_key`, `text_sample_redacted`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 25. `individual_comments_2016_18_text_fields.csv`
- **Exact File Path**: `data/interim/individual_comments_2016_18_text_fields.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 7,119 bytes
- **Row Count**: 19 | **Columns**: 10
- **Column Names**: `column_name`, `non_empty_responses`, `missing_responses`, `avg_length`, `max_length`, `unique_values`, `field_type`, `policy_relevance_recommendation`, `representative_samples_redacted`, `dataset_name`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 26. `policylens_sentiment_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/policylens_sentiment_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 1,246 bytes
- **Row Count**: 4 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 24 total nulls (`source_record_id`: 4, `consultation_id`: 4, `question_id`: 4, `question_text`: 4, `language`: 4)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `original_text`: 2 groups, `cleaned_text`: 2 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Label Distribution**: {'negative': 2, 'neutral': 2}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 27. `raw_file_checksums.csv`
- **Exact File Path**: `data/interim/raw_file_checksums.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 1,480 bytes
- **Row Count**: 10 | **Columns**: 3
- **Column Names**: `dataset_name`, `relative_path`, `sha256`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 28. `cross_dataset_row_id_overlap_review.csv`
- **Exact File Path**: `data/interim/review/cross_dataset_row_id_overlap_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 100,208 bytes
- **Row Count**: 145 | **Columns**: 25
- **Column Names**: `overlap_pair_id`, `shared_row_id`, `individual_source_row_number`, `compilation_source_row_number`, `individual_event`, `compilation_event`, `individual_source`, `compilation_source` ... (+17 more)
- **Missing Values**: 580 total nulls (`review_status`: 145, `review_decision`: 145, `review_notes`: 145, `reviewer`: 145)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `compilation_text_original`: 3 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 29. `duplicate_record_review.csv`
- **Exact File Path**: `data/interim/review/duplicate_record_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 5,297 bytes
- **Row Count**: 14 | **Columns**: 32
- **Column Names**: `duplicate_record_group_id`, `dataset_name`, `source_dataset`, `source_file`, `source_row_number`, `duplicate_text_group_id`, `exclusion_reason`, `row_rang_e` ... (+24 more)
- **Missing Values**: 336 total nulls (`exclusion_reason`: 14, `row_rang_e`: 14, `event`: 14, `v_nement`: 14, `source`: 14)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `duplicate_text_group_id`: 1 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 30. `exact_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/review/exact_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 225,891 bytes
- **Row Count**: 596 | **Columns**: 16
- **Column Names**: `dataset_name`, `source_row_index`, `source_row_number`, `column_name`, `text_key`, `text_sample_redacted`, `duplicate_group_id`, `duplicate_group_size` ... (+8 more)
- **Missing Values**: 2,384 total nulls (`review_status`: 596, `review_decision`: 596, `review_notes`: 596, `reviewer`: 596)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `text_key`: 188 groups, `text_sample_redacted`: 188 groups, `original_text`: 188 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 31. `near_duplicate_review.csv`
- **Exact File Path**: `data/interim/review/near_duplicate_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 11,463 bytes
- **Row Count**: 19 | **Columns**: 16
- **Column Names**: `near_duplicate_pair_id`, `left_dataset`, `left_row`, `left_column`, `right_dataset`, `right_row`, `right_column`, `similarity` ... (+8 more)
- **Missing Values**: 76 total nulls (`review_status`: 19, `review_decision`: 19, `review_notes`: 19, `reviewer`: 19)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 32. `row_count_verification.csv`
- **Exact File Path**: `data/interim/row_count_verification.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 408 bytes
- **Row Count**: 10 | **Columns**: 5
- **Column Names**: `dataset_name`, `input_rows`, `source_clean_rows`, `canonical_rows`, `excluded_rows`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 33. `sentiment140_test_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/sentiment140_test_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 185 bytes
- **Row Count**: 0 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: English Twitter polarity dataset (1.6M tweets); domain mismatch with government policy consultations; useful solely for polarity baseline pretraining.

### 34. `sentiment140_train_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/sentiment140_train_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 6,486,579 bytes
- **Row Count**: 33,347 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 166,735 total nulls (`consultation_id`: 33,347, `question_id`: 33,347, `question_text`: 33,347, `language`: 33,347, `exclusion_reason`: 33,347)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: `original_text`: 9613 groups, `cleaned_text`: 9794 groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Label Distribution**: {'negative': 19380, 'positive': 13967}
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: English Twitter polarity dataset (1.6M tweets); domain mismatch with government policy consultations; useful solely for polarity baseline pretraining.

### 35. `sentimix_test_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/sentimix_test_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 185 bytes
- **Row Count**: 0 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 36. `sentimix_train_duplicate_text_review.csv`
- **Exact File Path**: `data/interim/sentimix_train_duplicate_text_review.csv`
- **Architectural Role**: `interim_audit_artifact`
- **File Size**: 185 bytes
- **Row Count**: 0 | **Columns**: 14
- **Column Names**: `record_id`, `source_dataset`, `source_record_id`, `consultation_id`, `question_id`, `question_text`, `original_text`, `cleaned_text` ... (+6 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **auxiliary_sentiment_experiments**
  - *Rationale*: Hindi-English code-mixed or multilingual social media sentiment; suitable for auxiliary transfer learning, not for policy consultation stance.

### 37. `data_inventory.csv`
- **Exact File Path**: `reports/data_inventory.csv`
- **Architectural Role**: `audit_inventory`
- **File Size**: 2,455 bytes
- **Row Count**: 3 | **Columns**: 21
- **Column Names**: `dataset_name`, `dataset_title`, `relative_path`, `file_size_bytes`, `encoding`, `delimiter`, `header_structure`, `sha256` ... (+13 more)
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable**
  - *Rationale*: Not a model training or retrieval corpus.

## 6. Reference Academic Benchmark Tables (HinEmo-9 Paper) (19 files)

### 1. `agreement_out_blind__confusion_primary.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/agreement_out_blind__confusion_primary.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 325 bytes
- **Row Count**: 9 | **Columns**: 10
- **Column Names**: `Unnamed: 0`, `Love`, `Joy`, `Anger`, `Sadness`, `Fear`, `Surprise`, `Nostalgia`, `Devotion`, `Neutral`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 2. `agreement_out_blind__no_majority_items.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/agreement_out_blind__no_majority_items.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 1,207 bytes
- **Row Count**: 9 | **Columns**: 3
- **Column Names**: `item_id`, `primary_vote_counts`, `text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 3. `agreement_out_blind__per_label_metrics.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/agreement_out_blind__per_label_metrics.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 729 bytes
- **Row Count**: 9 | **Columns**: 6
- **Column Names**: `label`, `cohen_kappa`, `spearman_rho`, `prevalence`, `difficult_rate`, `n_positive_judgments`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Label Distribution**: {'Love': 1, 'Joy': 1, 'Anger': 1, 'Sadness': 1, 'Fear': 1}
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 4. `agreement_out_gold__confusion_primary.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/agreement_out_gold__confusion_primary.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 376 bytes
- **Row Count**: 9 | **Columns**: 10
- **Column Names**: `Unnamed: 0`, `Love`, `Joy`, `Anger`, `Sadness`, `Fear`, `Surprise`, `Nostalgia`, `Devotion`, `Neutral`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 5. `agreement_out_gold__no_majority_items.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/agreement_out_gold__no_majority_items.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 3,024 bytes
- **Row Count**: 23 | **Columns**: 3
- **Column Names**: `item_id`, `primary_vote_counts`, `text`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 6. `agreement_out_gold__per_label_metrics.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/agreement_out_gold__per_label_metrics.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 737 bytes
- **Row Count**: 9 | **Columns**: 6
- **Column Names**: `label`, `cohen_kappa`, `spearman_rho`, `prevalence`, `difficult_rate`, `n_positive_judgments`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Label Distribution**: {'Love': 1, 'Joy': 1, 'Anger': 1, 'Sadness': 1, 'Fear': 1}
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 7. `paper_tables__appendix_b.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__appendix_b.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 413 bytes
- **Row Count**: 9 | **Columns**: 4
- **Column Names**: `Unnamed: 0`, `English`, `Hindi`, `Hinglish`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 8. `paper_tables__bootstrap_ci.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__bootstrap_ci.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 243 bytes
- **Row Count**: 5 | **Columns**: 6
- **Column Names**: `model`, `seed_used`, `macro_f1`, `ci_low`, `ci_high`, `ci_width`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 9. `paper_tables__cooccurrence.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__cooccurrence.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 355 bytes
- **Row Count**: 9 | **Columns**: 10
- **Column Names**: `Unnamed: 0`, `Love`, `Joy`, `Anger`, `Sadness`, `Fear`, `Surprise`, `Nostalgia`, `Devotion`, `Neutral`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 10. `paper_tables__cooccurrence_human.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__cooccurrence_human.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 291 bytes
- **Row Count**: 9 | **Columns**: 10
- **Column Names**: `Unnamed: 0`, `Love`, `Joy`, `Anger`, `Sadness`, `Fear`, `Surprise`, `Nostalgia`, `Devotion`, `Neutral`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 11. `paper_tables__human_ceiling.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__human_ceiling.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 192 bytes
- **Row Count**: 3 | **Columns**: 6
- **Column Names**: `scope`, `items`, `judgements_used`, `judgements_skipped_no_reference`, `human_macro_f1`, `human_accuracy`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 12. `paper_tables__human_ceiling_by_language.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__human_ceiling_by_language.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 185 bytes
- **Row Count**: 4 | **Columns**: 6
- **Column Names**: `scope`, `items`, `judgements_used`, `skipped`, `macro_f1`, `accuracy`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 13. `paper_tables__lexical_correlates.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__lexical_correlates.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 1,905 bytes
- **Row Count**: 72 | **Columns**: 5
- **Column Names**: `scope`, `emotion`, `token`, `z`, `count_in_emotion`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 14. `paper_tables__lexical_correlates_by_lang.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__lexical_correlates_by_lang.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 6,290 bytes
- **Row Count**: 216 | **Columns**: 5
- **Column Names**: `scope`, `emotion`, `token`, `z`, `count_in_emotion`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 15. `paper_tables__significance.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__significance.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 668 bytes
- **Row Count**: 10 | **Columns**: 9
- **Column Names**: `model_a`, `model_b`, `mean_a`, `mean_b`, `diff`, `t_stat`, `p_ttest`, `p_wilcoxon`, `significant_05`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 16. `paper_tables__source_diversity.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__source_diversity.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 333 bytes
- **Row Count**: 4 | **Columns**: 11
- **Column Names**: `comments`, `distinct_videos`, `distinct_channels`, `distinct_genres`, `median_comments_per_video`, `max_comments_per_video`, `pct_from_top_10_videos`, `pct_from_top_10_channels` ... (+3 more)
- **Missing Values**: 23 total nulls (`distinct_videos`: 3, `distinct_channels`: 3, `distinct_genres`: 3, `median_comments_per_video`: 3, `max_comments_per_video`: 3)
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 17. `paper_tables__table_main.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__table_main.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 480 bytes
- **Row Count**: 5 | **Columns**: 7
- **Column Names**: `model`, `n_seeds`, `macro_f1`, `accuracy`, `English`, `Hindi`, `Hinglish`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 18. `paper_tables__table_per_cell.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__table_per_cell.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 6,717 bytes
- **Row Count**: 135 | **Columns**: 6
- **Column Names**: `model`, `emotion`, `language`, `n`, `f1`, `reliable`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Language Distribution**: {'English': 45, 'Hindi': 45, 'Hinglish': 45}
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.

### 19. `paper_tables__table_per_emotion.csv`
- **Exact File Path**: `hinemo9-v1.0.1/im-rakeshsahani-hinemo9-764fb36/reference/paper_tables__table_per_emotion.csv`
- **Architectural Role**: `reference_benchmark`
- **File Size**: 785 bytes
- **Row Count**: 5 | **Columns**: 10
- **Column Names**: `model`, `Love`, `Joy`, `Anger`, `Sadness`, `Fear`, `Surprise`, `Nostalgia`, `Devotion`, `Neutral`
- **Missing Values**: 0 null values
- **Blank Text Cells (Whitespace-only)**: 0
- **Exact Duplicate Rows**: 0
- **Duplicate Text Groups**: 0 duplicate text groups
- **Corresponding Cleaned Version**: None (Self is cleaned, annotation, or diagnostic artifact)
- **Cleaning Verification**: [INFERRED]
  - **Verification Source**: Inferred from filename only
- **Task Suitability**: **currently_unsuitable (audit/diagnostic/reference artifact)**
  - *Rationale*: Diagnostic, mapping, or academic paper reference table; not a model training corpus.
