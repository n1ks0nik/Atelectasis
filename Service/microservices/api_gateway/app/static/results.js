/**
 * Скрипт для страницы результатов
 */

let currentPage = 1;
let totalPages = 1;
let studies = [];
let selectedStudyId = null;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Проверяем URL параметры
    const urlParams = new URLSearchParams(window.location.search);
    const studyId = urlParams.get('study_id');

    if (studyId) {
        selectedStudyId = studyId;
    }

    loadResults();

    // Автообновление каждые 10 секунд если есть исследования в обработке
    setInterval(checkForProcessingStudies, 10000);
});

/**
 * Загрузка результатов
 */
async function loadResults(page = 1) {
    try {
        showLoading();

        const limit = 20;
        const offset = (page - 1) * limit;

        const response = await authorizedFetch(`/api/studies?limit=${limit}&offset=${offset}`);
        const data = await response.json();

        studies = data.studies || [];
        currentPage = page;
        totalPages = Math.ceil((data.total || studies.length) / limit);

        updateResultsTable();
        updatePagination();

        // Если есть выбранное исследование, показываем его детали
        if (selectedStudyId) {
            const study = studies.find(s => s.study_id === selectedStudyId);
            if (study) {
                showStudyDetails(selectedStudyId);
            }
            selectedStudyId = null; // Сбрасываем после показа
        }

    } catch (error) {
        console.error('Error loading results:', error);
        showError('Ошибка загрузки результатов');
    }
}

/**
 * Показ индикатора загрузки
 */
function showLoading() {
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = `
        <tr>
            <td colspan="7" class="loading">
                <i class="fas fa-spinner fa-spin"></i>
                Загрузка результатов...
            </td>
        </tr>
    `;
}

/**
 * Показ ошибки
 */
function showError(message) {
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = `
        <tr>
            <td colspan="7" class="text-center" style="color: var(--error-color); padding: 2rem;">
                <i class="fas fa-exclamation-triangle"></i>
                ${message}
            </td>
        </tr>
    `;
}

/**
 * Обновление таблицы результатов
 */
function updateResultsTable() {
    const tbody = document.getElementById('results-tbody');

    if (studies.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-muted" style="padding: 2rem;">
                    <i class="fas fa-inbox"></i><br>
                    Нет результатов для отображения
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = studies.map(study => {
        const statusClass = getStatusClass(study.status);
        const statusIcon = getStatusIcon(study.status);
        const probability = study.atelectasis_probability
            ? `${(study.atelectasis_probability * 100).toFixed(1)}%`
            : '-';

        const resultText = getResultText(study);

        return `
            <tr class="study-row" data-study-id="${study.study_id}">
                <td>
                    <code class="study-id">${study.study_id.substring(0, 8)}...</code>
                    <button class="btn-small" onclick="copyToClipboard('${study.study_id}')">
                        <i class="fas fa-copy"></i>
                    </button>
                </td>
                <td>
                    <div class="filename">
                        <i class="fas fa-file-medical"></i>
                        ${study.filename || 'Неизвестно'}
                    </div>
                </td>
                <td>
                    <span class="status-badge status-${statusClass}">
                        <i class="fas fa-${statusIcon}"></i>
                        ${getStatusText(study.status)}
                    </span>
                </td>
                <td>
                    <div class="datetime">
                        ${formatDateTime(study.created_at)}
                    </div>
                </td>
                <td>
                    <div class="result-summary">
                        ${resultText}
                    </div>
                </td>
                <td class="text-center">
                    <div class="probability ${getProbabilityClass(study.atelectasis_probability)}">
                        ${probability}
                    </div>
                </td>
                <td>
                    <div class="action-buttons">
                        <button class="btn btn-outline btn-small" onclick="showStudyDetails('${study.study_id}')">
                            <i class="fas fa-eye"></i>
                        </button>
                        ${study.status === 'completed' ? `
                            <button class="btn btn-outline btn-small" onclick="downloadStudyReports('${study.study_id}')">
                                <i class="fas fa-download"></i>
                            </button>
                        ` : ''}
                    </div>
                </td>
            </tr>
        `;
    }).join('');

    // Добавляем обработчики кликов для строк
    document.querySelectorAll('.study-row').forEach(row => {
        row.addEventListener('click', (e) => {
            // Игнорируем клики по кнопкам
            if (e.target.closest('button')) return;

            const studyId = row.dataset.studyId;
            showStudyDetails(studyId);
        });
    });
}

/**
 * Получение текста результата
 */
function getResultText(study) {
    if (study.status === 'completed' && study.conclusion) {
        return study.conclusion.substring(0, 50) + (study.conclusion.length > 50 ? '...' : '');
    } else if (study.status === 'processing') {
        return 'Анализ в процессе...';
    } else if (study.status === 'error') {
        return 'Ошибка обработки';
    } else {
        return '-';
    }
}

/**
 * Получение CSS класса для вероятности
 */
function getProbabilityClass(probability) {
    if (!probability) return '';

    const percent = probability * 100;
    if (percent >= 70) return 'high-risk';
    if (percent >= 30) return 'medium-risk';
    return 'low-risk';
}

/**
 * Копирование в буфер обмена
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('ID скопирован в буфер обмена', 'success');
    } catch (error) {
        console.error('Failed to copy:', error);
        showNotification('Не удалось скопировать', 'error');
    }
}

/**
 * Обновление пагинации
 */
function updatePagination() {
    const pageInfo = document.getElementById('page-info');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    pageInfo.textContent = `Страница ${currentPage} из ${totalPages}`;

    prevBtn.disabled = currentPage <= 1;
    nextBtn.disabled = currentPage >= totalPages;
}

/**
 * Загрузка страницы
 */
function loadPage(page) {
    if (page < 1 || page > totalPages) return;
    loadResults(page);
}

/**
 * Применение фильтров
 */
function applyFilters() {
    // В реальной системе здесь был бы запрос к серверу с фильтрами
    // Для демо просто перезагружаем данные
    loadResults(1);
    showNotification('Фильтры применены', 'info');
}

/**
 * Обновление результатов
 */
function refreshResults() {
    loadResults(currentPage);
    showNotification('Результаты обновлены', 'success');
}

/**
 * Показ деталей исследования
 */
async function showStudyDetails(studyId) {
    try {
        const response = await authorizedFetch(`/result/${studyId}`);
        const study = await response.json();

        showStudyModal(study);

    } catch (error) {
        console.error('Error loading study details:', error);
        showNotification('Ошибка загрузки деталей исследования', 'error');
    }
}

/**
 * Показ модального окна с деталями исследования
 */
function showStudyModal(study) {
    const modal = document.getElementById('result-modal');
    const modalBody = document.getElementById('modal-body');
    const downloadBtn = document.getElementById('download-btn');

    // Формируем содержимое модального окна
    let content = `
        <div class="study-details">
            <div class="detail-section">
                <h4><i class="fas fa-info-circle"></i> Основная информация</h4>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>ID исследования:</label>
                        <span><code>${study.study_id}</code></span>
                    </div>
                    <div class="detail-item">
                        <label>Файл:</label>
                        <span>${study.filename || '-'}</span>
                    </div>
                    <div class="detail-item">
                        <label>Статус:</label>
                        <span class="status-badge status-${getStatusClass(study.status)}">
                            <i class="fas fa-${getStatusIcon(study.status)}"></i>
                            ${getStatusText(study.status)}
                        </span>
                    </div>
                    <div class="detail-item">
                        <label>Создано:</label>
                        <span>${formatDateTime(study.created_at)}</span>
                    </div>
                    <div class="detail-item">
                        <label>Обновлено:</label>
                        <span>${formatDateTime(study.updated_at)}</span>
                    </div>
                </div>
            </div>
    `;

    if (study.status === 'completed' && study.results) {
        const results = study.results;
        const probability = (results.atelectasis_probability * 100).toFixed(1);

        content += `
            <div class="detail-section">
                <h4><i class="fas fa-brain"></i> Результаты анализа ИИ</h4>
                <div class="analysis-results">
                    <div class="probability-display ${getProbabilityClass(results.atelectasis_probability)}">
                        <div class="probability-circle" id="probability-circle-${study.study_id}">
                            <span class="probability-value">${probability}%</span>
                            <span class="probability-label">Вероятность ателектаза</span>
                        </div>
                    </div>
                    
                    <div class="result-details">
                        <div class="detail-item">
                            <label>Статус анализа:</label>
                            <span class="result-status">${getAnalysisStatusText(results.status)}</span>
                        </div>
                        
                        ${results.processing_time ? `
                            <div class="detail-item">
                                <label>Время обработки:</label>
                                <span>${results.processing_time.toFixed(2)} сек</span>
                            </div>
                        ` : ''}
                        
                        ${results.location ? `
                            <div class="detail-item">
                                <label>Локализация:</label>
                                <span>${results.location}</span>
                            </div>
                        ` : ''}
                        
                        ${results.bbox && results.bbox.length === 4 ? `
                            <div class="detail-item">
                                <label>Координаты (DICOM):</label>
                                <span>x: ${results.bbox[0]}-${results.bbox[2]}, y: ${results.bbox[1]}-${results.bbox[3]}</span>
                            </div>
                        ` : ''}
                    </div>
                </div>
                
                ${results.conclusion ? `
                    <div class="conclusion-section">
                        <h5><i class="fas fa-stethoscope"></i> Заключение</h5>
                        <div class="conclusion-text">
                            ${results.conclusion}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        // Настраиваем кнопку скачивания
        downloadBtn.style.display = 'block';
        downloadBtn.onclick = () => downloadStudyReports(study.study_id);

    } else if (study.status === 'error') {
        content += `
            <div class="detail-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Информация об ошибке</h4>
                <div class="error-info">
                    <p>${study.error || 'Произошла неизвестная ошибка при обработке файла'}</p>
                </div>
            </div>
        `;

        downloadBtn.style.display = 'none';

    } else {
        content += `
            <div class="detail-section">
                <h4><i class="fas fa-clock"></i> Обработка в процессе</h4>
                <div class="processing-info">
                    <div class="processing-spinner">
                        <i class="fas fa-spinner fa-spin"></i>
                    </div>
                    <p>Файл находится в обработке. Результаты появятся автоматически после завершения анализа.</p>
                </div>
            </div>
        `;

        downloadBtn.style.display = 'none';
    }

    content += '</div>';

    modalBody.innerHTML = content;
    modal.classList.add('show');

    // Обновляем прогресс-круг после рендеринга
    if (study.status === 'completed' && study.results) {
        setTimeout(() => {
            updateProbabilityCircle(study.study_id, study.results.atelectasis_probability);
        }, 100);
    }

    // Real-time обновление для обрабатывающихся исследований
    if (study.status === 'processing') {
        startModalRealTimeUpdate(study.study_id);
    }
}

/**
 * Обновление прогресс-круга с анимацией
 */
function updateProbabilityCircle(studyId, probability) {
    const circle = document.getElementById(`probability-circle-${studyId}`);
    if (!circle) return;

    const percentage = probability * 100;

    // Определяем цвет на основе вероятности
    let color;
    if (percentage >= 70) {
        color = 'var(--error-color)';
    } else if (percentage >= 30) {
        color = 'var(--warning-color)';
    } else {
        color = 'var(--success-color)';
    }

    // Анимируем прогресс
    let currentPercentage = 0;
    const duration = 1500; // 1.5 секунды
    const startTime = Date.now();

    function animate() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function для плавной анимации
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        currentPercentage = percentage * easeOutCubic;

        // Обновляем стиль
        circle.style.background = `conic-gradient(from 0deg, ${color} 0%, ${color} ${currentPercentage}%, #e9ecef ${currentPercentage}%, #e9ecef 100%)`;

        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }

    animate();
}

/**
 * Real-time обновление модального окна
 */
function startModalRealTimeUpdate(studyId) {
    const updateInterval = setInterval(async () => {
        try {
            const response = await authorizedFetch(`/result/${studyId}`);
            const updatedStudy = await response.json();

            if (updatedStudy.status !== 'processing') {
                clearInterval(updateInterval);
                showStudyModal(updatedStudy);

                // Также обновляем таблицу
                loadResults(currentPage);
            }

        } catch (error) {
            console.error('Error updating study status:', error);
            clearInterval(updateInterval);
        }
    }, 3000);

    // Очищаем интервал при закрытии модального окна
    const modal = document.getElementById('result-modal');
    const originalClose = closeModal;
    window.closeModal = function() {
        clearInterval(updateInterval);
        originalClose();
    };
}

/**
 * Получение текста статуса анализа
 */
function getAnalysisStatusText(status) {
    switch (status) {
        case 'normal':
            return 'Норма - признаков ателектаза не обнаружено';
        case 'atelectasis_only':
            return 'Обнаружен ателектаз';
        case 'other_pathologies':
            return 'Обнаружены другие патологии';
        default:
            return 'Неопределено';
    }
}

/**
 * Закрытие модального окна
 */
function closeModal() {
    const modal = document.getElementById('result-modal');
    modal.classList.remove('show');
}

/**
 * Скачивание отчетов исследования
 */
async function downloadStudyReports(studyId) {
    try {
        const response = await authorizedFetch(`/download/reports/${studyId}`);

        if (response.ok) {
            // Создаем ссылку для скачивания
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${studyId}_reports.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            showNotification('Отчеты загружены', 'success');
        } else {
            throw new Error('Ошибка загрузки файлов');
        }

    } catch (error) {
        console.error('Error downloading reports:', error);
        showNotification('Ошибка загрузки отчетов', 'error');
    }
}

/**
 * Проверка исследований в обработке для автообновления
 */
function checkForProcessingStudies() {
    const processingStudies = studies.filter(s => s.status === 'processing');

    if (processingStudies.length > 0) {
        // Тихое обновление данных
        loadResults(currentPage);
    }
}

/**
 * Закрытие модального окна DICOM
 */
function closeDicomModal() {
    const modal = document.getElementById('dicom-modal');
    modal.classList.remove('show');
}

// Вспомогательные функции уже определены в dashboard.js
function getStatusClass(status) {
    switch (status) {
        case 'completed': return 'completed';
        case 'processing': return 'processing';
        case 'error': return 'error';
        default: return 'processing';
    }
}

function getStatusIcon(status) {
    switch (status) {
        case 'completed': return 'check-circle';
        case 'processing': return 'spinner fa-spin';
        case 'error': return 'exclamation-triangle';
        default: return 'clock';
    }
}

function getStatusText(status) {
    switch (status) {
        case 'completed': return 'Завершено';
        case 'processing': return 'В обработке';
        case 'error': return 'Ошибка';
        default: return 'Неизвестно';
    }
}

function formatDateTime(isoString) {
    if (!isoString) return '-';

    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return 'Только что';
    if (diffMins < 60) return `${diffMins} мин назад`;
    if (diffHours < 24) return `${diffHours} ч назад`;
    if (diffDays < 7) return `${diffDays} дн назад`;

    return date.toLocaleDateString('ru-RU', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}