# Repository Structure

```text
medical-image-classification/
  .gitignore
  README.md
  requirements.txt
  REPOSITORY_STRUCTURE.md
  MASTER_FINAL_SUMMARY.md
  FINAL_READINESS_REPORT.md
  backend/
    README.md
    config.py
    gradcam_service.py
    inference.py
    main.py
    preprocessing.py
    schemas.py
    static/
      .gitkeep
  frontend/
    README.md
    app.js
    index.html
    styles.css
    assets/
      logo.png
  models/
    efficientnetb0_final.keras
  notebooks/
    README.md
    training_notebook.ipynb
  presentation/
    answers.md
    defense_preparation.md
    likely_questions.md
    presentation_outline.md
    speaker_notes.md
  src/
    __init__.py
    config.py
    download_dataset.py
    eda_dataset.py
    final_model_comparison.py
    generate_final_submission_package.py
    generate_phase2_report.py
    gradcam.py
    gradcam_analysis.py
    inspect_dataset.py
    integration_e2e_validation.py
    models.py
    prepare_splits.py
    preprocessing.py
    test_fastapi_backend.py
    test_frontend_workflow.py
    train_custom_cnn.py
    train_efficientnetb0.py
    train_resnet50.py
    verify_phase4.py
```

## Submission Contents

- Root documentation: `README.md`, `MASTER_FINAL_SUMMARY.md`, `FINAL_READINESS_REPORT.md`, `REPOSITORY_STRUCTURE.md`
- Dependencies: `requirements.txt`
- Source code: `src/`
- Backend: `backend/`
- Frontend: `frontend/`
- Training notebook: `notebooks/training_notebook.ipynb`
- Presentation package: `presentation/`
- Final model: `models/efficientnetb0_final.keras`

## Removed Before Submission

- Python cache directories and bytecode files
- `.pytest_cache/`
- Temporary Grad-CAM screenshots and validation outputs
- Duplicate reports not requested for final submission
- Obsolete model artifacts from non-selected models
- Local virtual environment, dataset, and generated result folders
