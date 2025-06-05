/**
 * Скрипт для дашборда
 */

let statsData = null;
let recentStudies = [];

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();

    // Обновляем данные каждые 30 секунд
    setInterval(loadDashboardData, 30000);
});

/**
 * Загрузка всех данных дашборда
 */
async function loadDashboardData() {
    try {
        // Загружаем статистику и последние исследования параллельно
        await Promise.all([
            loadStatistics(),
            loadRecentStudies(),
            checkSystemHealth()
        ]);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showNotification('Ошибка загрузки данных дашборда', 'error');
    }
}

/**
 * Загрузка статистики
 */
async function loadStatistics() {
    try {
        const response = await authorizedFetch('/statistics');
        statsData = await response.json();

        updateStatisticsDisplay();
    } catch (error) {
        console.error('Error loading statistics:', error);
        throw error;
    }
}

/**
 * Обновление отображения статистики
 */
function updateStatisticsDisplay() {
    if (!statsData) return;

    // Основные счетчики
    updateCounter('total-studies', statsData.total_studies || 0);
    updateCounter('completed-studies', statsData.completed || 0);
    updateCounter('processing-studies', statsData.processing || 0);
    updateCounter('error-studies', statsData.errors || 0);

    // Дополнительные метрики
    const avgTime = statsData.avg_processing_time || 0;
    const avgProb = statsData.avg_atelectasis_probability || 0;

    updateMetric('avg-processing-time', avgTime.toFixed(1));

    // Анимация для счетчиков
    animateCounters();
}

/**
 * Обновление значения счетчика с анимацией
 */
function updateCounter(elementId, value) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const currentValue = parseInt(element.textContent) || 0;
    animateValue(element, currentValue, value, 1000);
}

/**
 * Обновление метрики
 */
function updateMetric(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    }
}

/**
 * Анимация значения числа
 */
function animateValue(element, start, end, duration) {
    if (start === end) return;

    const range = end - start;
    const startTime = Date.now();

    function updateValue() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (range * easeOutCubic));

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(updateValue);
        }
    }

    updateValue();
}

/**
 * Анимация для всех счетчиков
 */
function animateCounters() {
    const counters = document.querySelectorAll('.stat-content h3');
    counters.forEach(counter => {
        counter.style.transform = 'scale(1.1)';
        counter.style.transition = 'transform 0.3s ease';

        setTimeout(() => {
            counter.style.transform = 'scale(1)';
        }, 300);
    });
}

/**
 * Загрузка последних исследований
 */
async function loadRecentStudies() {
    try {
        const response = await authorizedFetch('/api/studies?limit=10');
        const data = await response.json();
        recentStudies = data.studies || [];

        updateRecentStudiesTable();
    } catch (error) {
        console.error('Error loading recent studies:', error);
        throw error;
    }
}

/**
 * Обновление таблицы последних исследований
 */
function updateRecentStudiesTable() {
    const tbody = document.getElementById('recent-studies-tbody');
    if (!tbody) return;

    if (recentStudies.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Нет данных</td></tr>';
        return;
    }

    tbody.innerHTML = recentStudies.map(study => {
        const statusClass = getStatusClass(study.status);
        const statusIcon = getStatusIcon(study.status);
        const probability = study.atelectasis_probability
            ? `${(study.atelectasis_probability * 100).toFixed(1)}%`
            : '-';

        return `
            <tr onclick="viewStudyDetails('${study.study_id}')" style="cursor: pointer;">
                <td>
                    <code>${study.study_id.substring(0, 8)}...</code>
                </td>
                <td>${study.filename || '-'}</td>
                <td>
                    <span class="status-badge status-${statusClass}">
                        <i class="fas fa-${statusIcon}"></i>
                        ${getStatusText(study.status)}
                    </span>
                </td>
                <td>${formatDateTime(study.created_at)}</td>
                <td class="text-center">
                    ${study.status === 'completed' ? probability : '-'}
                </td>
            </tr>
        `;
    }).join('');
}

/**
 * Получение CSS класса для статуса
 */
function getStatusClass(status) {
    switch (status) {
        case 'completed': return 'completed';
        case 'processing': return 'processing';
        case 'error': return 'error';
        default: return 'processing';
    }
}

/**
 * Получение иконки для статуса
 */
function getStatusIcon(status) {
    switch (status) {
        case 'completed': return 'check-circle';
        case 'processing': return 'spinner fa-spin';
        case 'error': return 'exclamation-triangle';
        default: return 'clock';
    }
}

/**
 * Получение текста статуса
 */
function getStatusText(status) {
    switch (status) {
        case 'completed': return 'Завершено';
        case 'processing': return 'В обработке';
        case 'error': return 'Ошибка';
        default: return 'Неизвестно';
    }
}

/**
 * Форматирование даты и времени
 */
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

/**
 * Проверка состояния системы
 */
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const health = await response.json();

        updateSystemStatus(health);
    } catch (error) {
        console.error('Error checking system health:', error);
        updateSystemStatus({ status: 'error', kafka_connected: false });
    }
}

/**
 * Обновление отображения статуса системы
 */
function updateSystemStatus(health) {
    const statusElement = document.getElementById('system-status');
    if (!statusElement) return;

    const indicator = statusElement.querySelector('.status-indicator');
    const text = statusElement.querySelector('.status-text');

    if (health.status === 'healthy' && health.kafka_connected) {
        indicator.className = 'status-indicator';
        text.textContent = 'Система работает';
        text.style.color = 'var(--success-color)';
    } else {
        indicator.className = 'status-indicator error';
        text.textContent = 'Проблемы с системой';
        text.style.color = 'var(--error-color)';
    }
}

/**
 * Просмотр деталей исследования
 */
function viewStudyDetails(studyId) {
    // Переходим на страницу результатов с фокусом на конкретном исследовании
    window.location.href = `/results?study_id=${studyId}`;
}

/**
 * Обновление статистики (вызывается кнопкой)
 */
async function refreshStats() {
    const button = event.target;
    const originalText = button.innerHTML;

    // Показываем анимацию загрузки
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Обновление...';
    button.disabled = true;

    try {
        await loadDashboardData();
        showNotification('Статистика обновлена', 'success');
    } catch (error) {
        showNotification('Ошибка обновления статистики', 'error');
    } finally {
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

/**
 * Real-time обновления для активных исследований
 */
function startRealTimeUpdates() {
    // Проверяем наличие исследований в обработке
    const processingStudies = recentStudies.filter(s => s.status === 'processing');

    if (processingStudies.length > 0) {
        // Если есть исследования в обработке, обновляем чаще
        setTimeout(loadDashboardData, 5000);
    }
}

/**
 * Анимация для карточек статистики при наведении
 */
document.addEventListener('DOMContentLoaded', function() {
    const statCards = document.querySelectorAll('.stat-card');

    statCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(-5px) scale(1)';
        });
    });
});

/**
 * Показ детальной информации о метрике при клике
 */
function showMetricDetails(metricType) {
    let title, content;

    switch (metricType) {
        case 'processing_time':
            title = 'Время обработки';
            content = `
                <p>Среднее время обработки одного DICOM файла: <strong>${statsData?.avg_processing_time?.toFixed(2) || 0} секунд</strong></p>
                <p>Требования системы: ≤ 5 секунд</p>
                <p>Включает время на:</p>
                <ul>
                    <li>Валидацию DICOM файла</li>
                    <li>Предобработку изображения</li>
                    <li>Анализ нейронной сетью</li>
                    <li>Генерацию отчета</li>
                </ul>
            `;
            break;

        case 'atelectasis_prob':
            title = 'Вероятность ателектаза';
            content = `
                <p>Средняя вероятность обнаружения ателектаза: <strong>${((statsData?.avg_atelectasis_probability || 0) * 100).toFixed(1)}%</strong></p>
                <p>Классификация результатов:</p>
                <ul>
                    <li><strong>≥ 70%</strong> - Высокая вероятность ателектаза</li>
                    <li><strong>30-69%</strong> - Возможны другие патологии</li>
                    <li><strong>< 30%</strong> - Норма</li>
                </ul>
                <p><em>Все результаты требуют подтверждения врача</em></p>
            `;
            break;

        default:
            return;
    }

    showInfoModal(title, content);
}

/**
 * Показ информационного модального окна
 */
function showInfoModal(title, content) {
    // Создаем модальное окно
    const modal = document.createElement('div');
    modal.className = 'modal show';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3><i class="fas fa-info-circle"></i> ${title}</h3>
                <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
            </div>
            <div class="modal-body">
                ${content}
            </div>
            <div class="modal-footer">
                <button class="btn btn-outline" onclick="this.closest('.modal').remove()">Закрыть</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Закрытие по клику вне модального окна
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
}