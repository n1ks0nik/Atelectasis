/**
 * Система аутентификации
 */

// Проверка авторизации при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Если это страница входа, пропускаем проверку
    if (window.location.pathname === '/login') {
        return;
    }

    // Проверяем наличие токена
    const token = localStorage.getItem('auth_token');
    if (!token) {
        redirectToLogin();
        return;
    }

    // Проверяем валидность токена
    checkTokenValidity();
});

/**
 * Проверка валидности токена
 */
async function checkTokenValidity() {
    const token = localStorage.getItem('auth_token');
    if (!token) {
        redirectToLogin();
        return;
    }

    try {
        const response = await fetch('/health', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            // Токен недействителен
            logout();
        }
    } catch (error) {
        console.error('Error checking token validity:', error);
        // В случае ошибки сети, не выходим из системы
    }
}

/**
 * Получение токена из localStorage
 */
function getAuthToken() {
    return localStorage.getItem('auth_token');
}

/**
 * Получение данных пользователя
 */
function getUserData() {
    const userData = localStorage.getItem('user_data');
    return userData ? JSON.parse(userData) : null;
}

/**
 * Создание заголовков с авторизацией
 */
function getAuthHeaders() {
    const token = getAuthToken();
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };
}

/**
 * Выполнение авторизованного запроса
 */
async function authorizedFetch(url, options = {}) {
    const token = getAuthToken();

    if (!token) {
        redirectToLogin();
        throw new Error('No auth token');
    }

    const headers = {
        ...options.headers,
        'Authorization': `Bearer ${token}`
    };

    const response = await fetch(url, {
        ...options,
        headers
    });

    if (response.status === 401) {
        logout();
        throw new Error('Unauthorized');
    }

    return response;
}

/**
 * Выход из системы
 */
function logout() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    redirectToLogin();
}

/**
 * Перенаправление на страницу входа
 */
function redirectToLogin() {
    window.location.href = '/login';
}

/**
 * Отображение ошибки авторизации
 */
function showAuthError(message) {
    // Можно добавить тост-уведомления
    console.error('Auth error:', message);
    alert('Ошибка авторизации: ' + message);
}

/**
 * Форматирование ошибок от сервера
 */
function formatErrorMessage(error) {
    if (typeof error === 'string') {
        return error;
    }

    if (error.detail) {
        if (typeof error.detail === 'string') {
            return error.detail;
        }

        if (error.detail.message) {
            return error.detail.message;
        }
    }

    return 'Произошла неизвестная ошибка';
}

/**
 * Универсальная функция для обработки ошибок API
 */
async function handleApiResponse(response) {
    if (!response.ok) {
        let errorData;
        try {
            errorData = await response.json();
        } catch {
            errorData = { message: 'Ошибка сервера' };
        }

        const errorMessage = formatErrorMessage(errorData);
        throw new Error(errorMessage);
    }

    return response.json();
}

/**
 * Показать уведомление
 */
function showNotification(message, type = 'info') {
    // Создаем элемент уведомления
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;

    // Добавляем стили если их еще нет
    if (!document.getElementById('notification-styles')) {
        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                padding: 1rem;
                display: flex;
                align-items: center;
                gap: 1rem;
                z-index: 3000;
                max-width: 400px;
                animation: slideInRight 0.3s ease;
            }
            
            .notification-success { border-left: 4px solid #27ae60; }
            .notification-error { border-left: 4px solid #e74c3c; }
            .notification-warning { border-left: 4px solid #f39c12; }
            .notification-info { border-left: 4px solid #3498db; }
            
            .notification-content {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                flex: 1;
            }
            
            .notification-close {
                background: none;
                border: none;
                cursor: pointer;
                color: #666;
                padding: 0.25rem;
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(styles);
    }

    // Добавляем уведомление
    document.body.appendChild(notification);

    // Автоматически удаляем через 5 секунд
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-triangle';
        case 'warning': return 'exclamation-circle';
        default: return 'info-circle';
    }
}

/**
 * Обработка загрузки файлов с прогрессом
 */
async function uploadWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        // Обработчик прогресса
        if (onProgress) {
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    onProgress(percentComplete);
                }
            });
        }

        // Обработчик завершения
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    resolve(response);
                } catch (e) {
                    reject(new Error('Invalid JSON response'));
                }
            } else {
                try {
                    const error = JSON.parse(xhr.responseText);
                    reject(new Error(formatErrorMessage(error)));
                } catch (e) {
                    reject(new Error(`HTTP Error: ${xhr.status}`));
                }
            }
        });

        // Обработчик ошибок
        xhr.addEventListener('error', () => {
            reject(new Error('Network error'));
        });

        // Настройка запроса
        xhr.open('POST', url);

        // Добавляем авторизацию
        const token = getAuthToken();
        if (token) {
            xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        }

        // Отправляем запрос
        xhr.send(formData);
    });
}