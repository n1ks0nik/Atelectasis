import os
import requests
import time
import json
from pathlib import Path
import base64

# Конфигурация
API_URL = "http://localhost:8000"  # Адрес API Gateway
TEST_TOKEN = "test-jwt-token-12345"  # Тестовый JWT токен

# Пути к тестовым файлам
# SCRIPT_DIR = Path(__file__).parent
TEST_PNG_PATH = "test.png"  # Путь к тестовому PNG
TEST_DICOM_PATH = "dicom.dcm"  # Путь к тестовому DICOM


def create_test_dicom_from_png():
    """
    Создает тестовый DICOM файл из PNG используя API
    """
    print("🔧 Creating test DICOM from PNG...")

    # Проверяем наличие PNG файла
    if not TEST_PNG_PATH.exists():
        print(f"❌ PNG file not found at {TEST_PNG_PATH}")
        print("Creating a dummy PNG for testing...")

        # Создаем простое тестовое изображение
        import numpy as np
        from PIL import Image

        # Создаем директорию если её нет
        TEST_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Создаем градиентное изображение 1024x1024 (имитация рентгена)
        img_array = np.zeros((1024, 1024), dtype=np.uint8)
        # Добавляем градиент и шум для реалистичности
        for i in range(1024):
            for j in range(1024):
                img_array[i, j] = int(255 * (0.3 + 0.4 * (i / 1024) + 0.3 * np.random.random()))

        # Добавляем "тень" в области легких
        img_array[300:700, 200:400] = img_array[300:700, 200:400] * 0.7
        img_array[300:700, 600:800] = img_array[300:700, 600:800] * 0.7

        img = Image.fromarray(img_array, mode='L')
        img.save(TEST_PNG_PATH)
        print(f"✅ Created test PNG at {TEST_PNG_PATH}")

    # Отправляем PNG для конвертации в DICOM
    with open(TEST_PNG_PATH, 'rb') as f:
        files = {'png_file': ('test_chest.png', f, 'image/png')}
        response = requests.post(
            f"{API_URL}/test/create_dicom",
            files=files,
            params={'add_patient_info': True}
        )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ DICOM created: {result}")
        return result.get('dicom_path')
    else:
        print(f"❌ Failed to create DICOM: {response.status_code}")
        print(response.text)
        return None


def test_health_check():
    """
    Проверка health endpoint
    """
    print("\n🏥 Testing health check...")

    response = requests.get(f"{API_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed:")
        print(f"   Status: {data['status']}")
        print(f"   Kafka connected: {data['kafka_connected']}")
        print(f"   Timestamp: {data['timestamp']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False


def test_analyze_dicom(dicom_path=None):
    """
    Тестирование основного endpoint анализа
    """
    print("\n🔍 Testing DICOM analysis...")

    # Если путь не указан, используем тестовый DICOM
    if not dicom_path:
        dicom_path = TEST_DICOM_PATH

    # Проверяем наличие файла
    if not Path(dicom_path).exists():
        print(f"❌ DICOM file not found at {dicom_path}")

        # Пытаемся создать из PNG
        print("Attempting to create DICOM from PNG...")
        created_path = create_test_dicom_from_png()
        if created_path:
            # Копируем созданный файл в нужное место
            import shutil
            TEST_DICOM_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(created_path, TEST_DICOM_PATH)
            dicom_path = TEST_DICOM_PATH
        else:
            return None

    # Отправляем DICOM на анализ
    with open(dicom_path, 'rb') as f:
        files = {'file': ('test.dcm', f, 'application/dicom')}
        headers = {'Authorization': f'Bearer {TEST_TOKEN}'}

        print(f"📤 Uploading {dicom_path}...")
        response = requests.post(
            f"{API_URL}/analyze",
            files=files,
            headers=headers
        )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Analysis started:")
        print(f"   Study ID: {result['study_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        return result['study_id']
    else:
        print(f"❌ Analysis failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None


def test_get_result(study_id):
    """
    Проверка получения результата
    """
    print(f"\n📊 Getting result for study_id: {study_id}")

    headers = {'Authorization': f'Bearer {TEST_TOKEN}'}

    # Делаем несколько попыток с задержкой
    max_attempts = 30
    for attempt in range(max_attempts):
        response = requests.get(
            f"{API_URL}/result/{study_id}",
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()

            if result.get('status') == 'completed':
                print(f"✅ Analysis completed!")
                print(f"   Results: {json.dumps(result, indent=2)}")
                return True
            elif result.get('status') == 'processing':
                print(f"⏳ Still processing... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)
            else:
                print(f"❓ Unexpected status: {result.get('status')}")
                print(f"   Full response: {json.dumps(result, indent=2)}")
                time.sleep(2)
        else:
            print(f"❌ Failed to get result: {response.status_code}")
            print(response.text)
            return False

    print("⏱️ Timeout waiting for result")
    return False


def test_storage_service(study_id):
    """
    Проверка работы storage service через проверку созданных файлов
    """
    print("\n💾 Testing Storage Service functionality...")

    # Даем время на обработку storage service
    print("⏳ Waiting for storage service to process results...")
    time.sleep(3)

    # Проверяем создание отчетов
    reports_dir = Path("./reports")  # Путь должен соответствовать REPORTS_DIR в storage service

    checks = {
        "JSON Report": reports_dir / "json" / f"{study_id}_report.json",
        "API JSON": reports_dir / "json_api" / f"{study_id}_api.json",
        "DICOM SR": reports_dir / "dicom_sr" / f"{study_id}_sr.dcm"
    }

    all_passed = True

    for report_type, file_path in checks.items():
        if file_path.exists():
            print(f"✅ {report_type} created: {file_path}")

            # Для JSON файлов показываем содержимое
            if file_path.suffix == '.json':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if report_type == "JSON Report":
                        print(f"   - Status: {data.get('status')}")
                        print(f"   - Atelectasis probability: {data.get('atelectasis_probability', 0):.2%}")
                        if data.get('bbox'):
                            print(f"   - Bounding box: {data.get('bbox')}")
                        print(f"   - Conclusion: {data.get('conclusion', 'N/A')[:100]}...")

                    elif report_type == "API JSON":
                        print(f"   - API version: {data.get('api_version')}")
                        print(f"   - Report type: {data.get('report_type')}")
                        if data.get('localization'):
                            loc = data['localization']
                            print(
                                f"   - Localization: [{loc.get('xmin')}, {loc.get('ymin')}, {loc.get('xmax')}, {loc.get('ymax')}]")

                except Exception as e:
                    print(f"   ⚠️ Error reading {report_type}: {e}")

            elif report_type == "DICOM SR":
                print(f"   - File size: {file_path.stat().st_size} bytes")
        else:
            print(f"❌ {report_type} NOT found at: {file_path}")
            all_passed = False

            # Проверяем, существует ли директория
            if not file_path.parent.exists():
                print(f"   ⚠️ Directory doesn't exist: {file_path.parent}")

    # Проверяем логи storage service
    storage_log_indicators = [
        "JSON report saved",
        "API JSON report generated",
        "Result successfully stored"
    ]

    print("\n📋 Storage Service indicators to check in logs:")
    for indicator in storage_log_indicators:
        print(f"   - Look for: '{indicator}'")

    return all_passed


def test_multiple_studies():
    """
    Тест с несколькими исследованиями для проверки хранилища
    """
    print("\n🔄 Testing multiple studies processing...")

    study_ids = []

    # Создаем 3 исследования
    for i in range(3):
        print(f"\n--- Study {i + 1}/3 ---")
        study_id = test_analyze_dicom()
        if study_id:
            study_ids.append(study_id)
            print(f"✅ Study {i + 1} started: {study_id}")
            time.sleep(2)  # Небольшая задержка между запросами
        else:
            print(f"❌ Failed to start study {i + 1}")

    if not study_ids:
        print("❌ No studies were created successfully")
        return

    print(f"\n📊 Created {len(study_ids)} studies, waiting for processing...")
    time.sleep(10)  # Даем время на обработку

    # Проверяем результаты всех исследований
    successful_studies = []
    for study_id in study_ids:
        print(f"\n--- Checking study: {study_id} ---")
        headers = {'Authorization': f'Bearer {TEST_TOKEN}'}
        response = requests.get(f"{API_URL}/result/{study_id}", headers=headers)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Got result for {study_id}: {result.get('status')}")
            if result.get('status') == 'completed':
                successful_studies.append(study_id)
        else:
            print(f"❌ Failed to get result for {study_id}")

    print(f"\n📈 Summary: {len(successful_studies)}/{len(study_ids)} studies completed successfully")

    # Проверяем файлы для всех успешных исследований
    if successful_studies:
        print("\n💾 Checking storage for all successful studies...")
        for study_id in successful_studies:
            test_storage_service(study_id)


def run_full_test():
    """
    Запуск полного теста системы
    """
    print("🚀 Starting full system test...\n")

    # 1. Проверка health
    if not test_health_check():
        print("❌ System is not healthy, stopping tests")
        return

    # 2. Анализ DICOM
    study_id = test_analyze_dicom()
    if not study_id:
        print("❌ Failed to start analysis, stopping tests")
        return

    # 3. Ожидание и получение результата
    print("\n⏳ Waiting for processing to complete...")
    time.sleep(5)

    if test_get_result(study_id):
        print("\n✅ Analysis completed successfully!")

        # 4. Проверка storage service
        if test_storage_service(study_id):
            print("\n✅ Storage Service test passed!")
        else:
            print("\n⚠️ Storage Service test partially failed")
    else:
        print("\n❌ Analysis failed")


def test_invalid_file():
    """
    Тест с невалидным файлом
    """
    print("\n🧪 Testing invalid file upload...")

    # Создаем текстовый файл вместо DICOM
    with open("test.txt", "w") as f:
        f.write("This is not a DICOM file")

    try:
        with open("test.txt", 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            headers = {'Authorization': f'Bearer {TEST_TOKEN}'}

            response = requests.post(
                f"{API_URL}/analyze",
                files=files,
                headers=headers
            )

        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 400:
            print("✅ Invalid file correctly rejected")
        else:
            print("❌ Invalid file should have been rejected")

    finally:
        # Удаляем временный файл
        if os.path.exists("test.txt"):
            os.remove("test.txt")


def test_missing_auth():
    """
    Тест без авторизации
    """
    print("\n🔐 Testing missing authentication...")

    # Создаем dummy файл
    with open("dummy.dcm", "wb") as f:
        f.write(b"DUMMY")

    try:
        with open("dummy.dcm", 'rb') as f:
            files = {'file': ('dummy.dcm', f, 'application/dicom')}
            # Не отправляем заголовок Authorization
            response = requests.post(
                f"{API_URL}/analyze",
                files=files
            )

        print(f"Response status: {response.status_code}")

        if response.status_code == 403 or response.status_code == 401:
            print("✅ Unauthorized request correctly rejected")
        else:
            print("❌ Unauthorized request should have been rejected")
            print(f"Response: {response.text}")

    finally:
        if os.path.exists("dummy.dcm"):
            os.remove("dummy.dcm")


def check_system_statistics():
    """
    Проверка общей статистики системы (если реализован соответствующий endpoint)
    """
    print("\n📊 Checking system statistics...")

    # Это предполагает, что в будущем будет добавлен endpoint для статистики
    # Пока просто проверяем наличие файлов отчетов

    reports_dir = Path("./reports")

    if reports_dir.exists():
        json_count = len(list((reports_dir / "json").glob("*.json"))) if (reports_dir / "json").exists() else 0
        api_json_count = len(list((reports_dir / "json_api").glob("*.json"))) if (
                    reports_dir / "json_api").exists() else 0
        dicom_sr_count = len(list((reports_dir / "dicom_sr").glob("*.dcm"))) if (
                    reports_dir / "dicom_sr").exists() else 0

        print(f"📁 Storage statistics:")
        print(f"   - JSON reports: {json_count}")
        print(f"   - API JSON reports: {api_json_count}")
        print(f"   - DICOM SR files: {dicom_sr_count}")

        # Проверяем консистентность
        if json_count == api_json_count:
            print("✅ Report counts are consistent")
        else:
            print("⚠️ Report counts mismatch - possible processing issues")
    else:
        print("❌ Reports directory not found")


def interactive_menu():
    """
    Интерактивное меню для тестирования
    """
    while True:
        print("\n" + "=" * 50)
        print("🧪 ATELECTASIS DETECTION SYSTEM TEST MENU")
        print("=" * 50)
        print("1. Run full test suite (including storage)")
        print("2. Test health check only")
        print("3. Test DICOM analysis")
        print("4. Test invalid file handling")
        print("5. Test missing authentication")
        print("6. Create test DICOM from PNG")
        print("7. Test multiple studies")
        print("8. Check system statistics")
        print("9. Test storage service only (requires study_id)")
        print("0. Exit")
        print("=" * 50)

        choice = input("Select option: ")

        if choice == "1":
            run_full_test()
        elif choice == "2":
            test_health_check()
        elif choice == "3":
            study_id = test_analyze_dicom()
            if study_id:
                input("\nPress Enter to check result...")
                if test_get_result(study_id):
                    input("\nPress Enter to test storage service...")
                    test_storage_service(study_id)
        elif choice == "4":
            test_invalid_file()
        elif choice == "5":
            test_missing_auth()
        elif choice == "6":
            create_test_dicom_from_png()
        elif choice == "7":
            test_multiple_studies()
        elif choice == "8":
            check_system_statistics()
        elif choice == "9":
            study_id = input("Enter study_id to check: ").strip()
            if study_id:
                test_storage_service(study_id)
            else:
                print("❌ Invalid study_id")
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Atelectasis Detection System')
    parser.add_argument('--auto', action='store_true', help='Run automatic full test')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API Gateway URL')
    parser.add_argument('--token', default='test-jwt-token-12345', help='JWT token for auth')
    parser.add_argument('--multi', action='store_true', help='Run multiple studies test')

    args = parser.parse_args()

    API_URL = args.api_url
    TEST_TOKEN = args.token

    if args.auto:
        run_full_test()
    elif args.multi:
        test_multiple_studies()
    else:
        interactive_menu()