/**
 * Скрипт для страницы загрузки файлов
 */

let selectedFile = null;
let currentStudyId = null;
let uploadInProgress = false;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadArea();
    initializePngConverter();
});

/**
 * Инициализация области загрузки
 */
function initializeUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    // Клик по области загрузки
    uploadArea.addEventListener('click', () => {
        if (!uploadInProgress) {
            fileInput.click();
        }
    });

    // Выбор файла через input
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelection(file);
        }
    });

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });
}

/**
 * Обработка выбора файла
 */
function handleFileSelection(file) {
    if (uploadInProgress) return;

    // Проверка типа файла
    if (!file.name.toLowerCase().endsWith('.dcm') && !file.name.toLowerCase().endsWith('.dicom')) {
        showNotification('Пожалуйста, выберите DICOM файл (.dcm)', 'error');
        return;
    }

    // Проверка размера файла (100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('Файл слишком большой. Максимальный размер: 100 МБ', 'error');
        return;
    }

    selectedFile = file;
    showFileInfo(file);
}

/**
 * Отображение информации о выбранном файле
 */
function showFileInfo(file) {
    const uploadArea = document.getElementById('upload-area');
    const fileInfo = document.getElementById('file-info');
    const uploadActions = document.getElementById('upload-actions');

    // Скрываем область загрузки
    uploadArea.style.display = 'none';

    // Показываем информацию о файле
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatFileSize(file.size);
    fileInfo.style.display = 'flex';
    uploadActions.style.display = 'block';
}

/**
 * Форматирование размера файла
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Очистка выбранного файла
 */
function clearFile() {
    if (uploadInProgress) return;

    selectedFile = null;

    const uploadArea = document.getElementById('upload-area');
    const fileInfo = document.getElementById('file-info');
    const uploadActions = document.getElementById('upload-actions');
    const fileInput = document.getElementById('file-input');

    uploadArea.style.display = 'block';
    fileInfo.style.display = 'none';
    uploadActions.style.display = 'none';
    fileInput.value = '';
}

/**
 * Загрузка файла
 */
async function uploadFile() {
    if (!selectedFile || uploadInProgress) return;

    uploadInProgress = true;

    try {
        // Показываем прогресс
        showUploadProgress();

        // Создаем FormData
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Загружаем файл с отслеживанием прогресса
        const response = await uploadWithProgress('/analyze', formData, updateUploadProgress);

        currentStudyId = response.study_id;

        // Показываем успех
        showUploadSuccess(response);

        // Начинаем отслеживание статуса
        startStatusTracking();

    } catch (error) {
        showUploadError(error.message);
    } finally {
        uploadInProgress = false;
    }
}

/**
 * Показ прогресса загрузки
 */
function showUploadProgress() {
    document.querySelector('.upload-container').style.display = 'none';
    document.getElementById('upload-progress').style.display = 'block';

    // Активируем первый шаг
    setProgressStep('upload', true);
}

/**
 * Обновление прогресса загрузки
 */
function updateUploadProgress(percent) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressFill.style.width = `${Math.min(percent, 25)}%`;

    if (percent < 100) {
        progressText.textContent = `Загрузка файла... ${Math.round(percent)}%`;
    } else {
        progressText.textContent = 'Валидация DICOM файла...';
        setProgressStep('validate', true);
        progressFill.style.width = '50%';
    }
}

/**
 * Установка состояния шага прогресса
 */
function setProgressStep(stepId, active = false, completed = false) {
    const step = document.getElementById(`step-${stepId}`);
    if (!step) return;

    step.classList.remove('active', 'completed');

    if (completed) {
        step.classList.add('completed');
    } else if (active) {
        step.classList.add('active');
    }
}

/**
 * Показ успешной загрузки
 */
function showUploadSuccess(response) {
    document.getElementById('study-id').textContent = response.study_id;
    document.getElementById('upload-progress').style.display = 'none';
    document.getElementById('upload-result').style.display = 'block';
    document.getElementById('result-success').style.display = 'block';

    showNotification('Файл успешно загружен и отправлен на анализ', 'success');
}

/**
 * Показ ошибки загрузки
 */
function showUploadError(errorMessage) {
    document.getElementById('error-message').textContent = errorMessage;
    document.getElementById('upload-progress').style.display = 'none';
    document.getElementById('upload-result').style.display = 'block';
    document.getElementById('result-error').style.display = 'block';

    showNotification('Ошибка при загрузке файла', 'error');
}

/**
 * Отслеживание статуса обработки
 */
function startStatusTracking() {
    if (!currentStudyId) return;

    const checkStatus = async () => {
        try {
            const response = await authorizedFetch(`/result/${currentStudyId}`);
            const result = await response.json();

            updateProgressFromStatus(result.status);

            if (result.status === 'completed') {
                showAnalysisComplete(result);
                return; // Останавливаем отслеживание
            } else if (result.status === 'error') {
                showUploadError(result.error || 'Ошибка при обработке файла');
                return; // Останавливаем отслеживание
            }

            // Продолжаем отслеживание
            setTimeout(checkStatus, 2000);

        } catch (error) {
            console.error('Error checking status:', error);
            setTimeout(checkStatus, 5000); // Увеличиваем интервал при ошибке
        }
    };

    // Начинаем проверку через 2 секунды
    setTimeout(checkStatus, 2000);
}

/**
 * Обновление прогресса на основе статуса
 */
function updateProgressFromStatus(status) {
    const progressText = document.getElementById('progress-text');
    const progressFill = document.getElementById('progress-fill');

    switch (status) {
        case 'processing':
            progressText.textContent = 'Анализ изображения нейронной сетью...';
            setProgressStep('upload', false, true);
            setProgressStep('validate', false, true);
            setProgressStep('analyze', true);
            progressFill.style.width = '75%';
            break;

        case 'completed':
            progressText.textContent = 'Анализ завершен!';
            setProgressStep('analyze', false, true);
            setProgressStep('complete', true);
            progressFill.style.width = '100%';
            break;
    }
}

/**
 * Показ завершения анализа
 */
function showAnalysisComplete(result) {
    // Обновляем сообщение об успехе
    const successDiv = document.getElementById('result-success');

    if (result.results && result.results.atelectasis_probability !== undefined) {
        const probability = (result.results.atelectasis_probability * 100).toFixed(1);

        successDiv.innerHTML = `
            <div class="result-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h3>Анализ завершен!</h3>
            <p>ID исследования: <span id="study-id">${currentStudyId}</span></p>
            <div class="analysis-summary">
                <h4>Результат анализа:</h4>
                <p><strong>Вероятность ателектаза:</strong> ${probability}%</p>
                <p><strong>Статус:</strong> ${getStatusDescription(result.results.status)}</p>
                ${result.results.conclusion ? `<p><strong>Заключение:</strong> ${result.results.conclusion}</p>` : ''}
            </div>
            <div class="result-actions">
                <button class="btn btn-primary" onclick="checkResult()">
                    <i class="fas fa-eye"></i>
                    Посмотреть подробности
                </button>
                <button class="btn btn-secondary" onclick="uploadAnother()">
                    <i class="fas fa-plus"></i>
                    Загрузить еще файл
                </button>
            </div>
        `;
    }

    showNotification('Анализ успешно завершен!', 'success');
}

/**
 * Получение описания статуса
 */
function getStatusDescription(status) {
    switch (status) {
        case 'normal':
            return 'Норма';
        case 'atelectasis_only':
            return 'Обнаружен ателектаз';
        case 'other_pathologies':
            return 'Обнаружены другие патологии';
        default:
            return 'Неопределено';
    }
}

/**
 * Проверка результата
 */
function checkResult() {
    if (currentStudyId) {
        window.location.href = `/results?study_id=${currentStudyId}`;
    }
}

/**
 * Загрузка еще одного файла
 */
function uploadAnother() {
    // Сброс всех состояний
    selectedFile = null;
    currentStudyId = null;
    uploadInProgress = false;

    // Скрываем результат
    document.getElementById('upload-result').style.display = 'none';
    document.getElementById('result-success').style.display = 'none';
    document.getElementById('result-error').style.display = 'none';
    document.getElementById('upload-progress').style.display = 'none';

    // Показываем форму загрузки
    document.querySelector('.upload-container').style.display = 'block';
    clearFile();
}

/**
 * Инициализация конвертера PNG в DICOM
 */
function initializePngConverter() {
    const pngInput = document.getElementById('png-input');
    const convertBtn = document.getElementById('convert-btn');

    pngInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            convertBtn.disabled = false;
            convertBtn.innerHTML = `<i class="fas fa-magic"></i> Создать DICOM из ${file.name}`;
        } else {
            convertBtn.disabled = true;
            convertBtn.innerHTML = '<i class="fas fa-magic"></i> Создать DICOM';
        }
    });
}

/**
 * Конвертация PNG в DICOM
 */
async function convertPngToDicom() {
    const pngInput = document.getElementById('png-input');
    const convertBtn = document.getElementById('convert-btn');
    const resultDiv = document.getElementById('conversion-result');

    if (!pngInput.files[0]) {
        showNotification('Выберите PNG файл для конвертации', 'warning');
        return;
    }

    const originalText = convertBtn.innerHTML;
    convertBtn.disabled = true;
    convertBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Создание DICOM...';

    try {
        const formData = new FormData();
        formData.append('png_file', pngInput.files[0]);
        formData.append('add_patient_info', 'true');

        const response = await fetch('/test/create_dicom', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            resultDiv.innerHTML = `
                <div style="margin-top: 1rem; padding: 1rem; background: #d4edda; border-radius: 4px; color: #155724;">
                    <i class="fas fa-check-circle"></i>
                    DICOM файл успешно создан: ${result.dicom_path}
                </div>
            `;
            showNotification('DICOM файл создан успешно', 'success');
        } else {
            throw new Error(result.detail || 'Ошибка создания DICOM');
        }

    } catch (error) {
        resultDiv.innerHTML = `
            <div style="margin-top: 1rem; padding: 1rem; background: #f8d7da; border-radius: 4px; color: #721c24;">
                <i class="fas fa-exclamation-triangle"></i>
                Ошибка: ${error.message}
            </div>
        `;
        showNotification('Ошибка создания DICOM файла', 'error');
    } finally {
        convertBtn.disabled = false;
        convertBtn.innerHTML = originalText;
    }
}

/**
 * Добавление стилей для результатов анализа
 */
const analysisStyles = `
    .analysis-summary {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: left;
    }
    
    .analysis-summary h4 {
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .analysis-summary p {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    .analysis-summary strong {
        color: var(--primary-color);
    }
`;

// Добавляем стили в head
if (!document.getElementById('analysis-styles')) {
    const styleElement = document.createElement('style');
    styleElement.id = 'analysis-styles';
    styleElement.textContent = analysisStyles;
    document.head.appendChild(styleElement);
}