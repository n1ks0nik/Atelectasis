-- init.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Таблица исследований
CREATE TABLE studies (
    study_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(255),
    study_instance_uid VARCHAR(255) UNIQUE,
    status VARCHAR(50) NOT NULL DEFAULT 'processing',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_token VARCHAR(500),
    filename VARCHAR(255),
    file_size BIGINT,
    CHECK (status IN ('processing', 'completed', 'error', 'storage_error'))
);

-- Таблица результатов анализа
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    study_id UUID REFERENCES studies(study_id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL,
    atelectasis_probability DECIMAL(5,4),
    processing_time DECIMAL(10,3),
    bbox_xmin INTEGER,
    bbox_ymin INTEGER,
    bbox_xmax INTEGER,
    bbox_ymax INTEGER,
    location_description VARCHAR(255),
    conclusion TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(study_id)
);

-- Таблица других патологий
CREATE TABLE other_pathologies (
    id SERIAL PRIMARY KEY,
    study_id UUID REFERENCES studies(study_id) ON DELETE CASCADE,
    other_pathologies_probability DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица путей к файлам отчетов
CREATE TABLE report_files (
    id SERIAL PRIMARY KEY,
    study_id UUID REFERENCES studies(study_id) ON DELETE CASCADE,
    file_type VARCHAR(50) NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (file_type IN ('json_report', 'api_json', 'dicom_sr', 'dicom_annotated', 'dicom_series'))
);

-- Индексы для быстрого поиска
CREATE INDEX idx_studies_status ON studies(status);
CREATE INDEX idx_studies_created_at ON studies(created_at);
CREATE INDEX idx_analysis_results_study_id ON analysis_results(study_id);
CREATE INDEX idx_report_files_study_id ON report_files(study_id);

-- Функция для обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Триггер для автообновления updated_at
CREATE TRIGGER update_studies_updated_at BEFORE UPDATE ON studies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();