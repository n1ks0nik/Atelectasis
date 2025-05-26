import os
import json
from datetime import datetime
from pydicom import dcmread
from pydicom.uid import generate_uid
from highdicom import ComprehensiveSRDocument
from highdicom.sr.content import ObservationContext, Finding, ImageRegion, RegionOfInterest
from highdicom.sr.coding import Code
from highdicom.enum import ContentQualificationValues, GraphicTypeValues


def generate_dicom_sr_from_json(json_path: str, dicom_path: str, output_path: str) -> bool:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
    except Exception as e:
        print(f"[✗] Ошибка при чтении JSON: {e}")
        return False

    try:
        ds = dcmread(dicom_path)
    except Exception as e:
        print(f"[✗] Ошибка при чтении DICOM: {e}")
        return False

    if not hasattr(ds, 'SOPClassUID') or not hasattr(ds, 'SOPInstanceUID'):
        print("[✗] DICOM не содержит необходимых UID.")
        return False

    status = report_data.get("status", "unknown")
    conclusion = report_data.get("conclusion", "No conclusion")
    bbox = report_data.get("bbox", [])

    findings = []
    if "atelectasis" in status.lower():
        findings.append(Finding(
            name=Code("67782005", "SCT", "Atelectasis"),
            description=conclusion
        ))

    # Формируем ROI (если bbox есть)
    roi = []
    if len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        roi.append(
            RegionOfInterest(
                graphic_type=GraphicTypeValues.RECTANGLE,
                graphic_data=[[x0, y0], [x1, y1]],
                image_region=ImageRegion(
                    referenced_sop_class_uid=ds.SOPClassUID,
                    referenced_sop_instance_uid=ds.SOPInstanceUID,
                    frame_number=1
                )
            )
        )

    try:
        sr_document = ComprehensiveSRDocument(
            evidence=[ds],
            observer_context=ObservationContext(),
            procedure_reported=Code("721981007", "SCT", "Chest X-ray"),
            title=Code("18748-4", "LN", "Radiology Report"),
            completion_flag="COMPLETE",
            verification_flag="UNVERIFIED",
            content_date=datetime.now().date(),
            content_time=datetime.now().time(),
            manufacturer="AtelectasisAI",
            series_instance_uid=generate_uid(),
            sop_instance_uid=generate_uid(),
            instance_number=1,
            content_qualification=ContentQualificationValues.RESEARCH,
            findings=findings,
            regions=roi,
            description=conclusion
        )
    except Exception as e:
        print(f"[✗] Ошибка создания SR документа: {e}")
        return False

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sr_document.save_as(output_path)
        print(f"[✓] DICOM SR успешно сохранён: {output_path}")
        return True
    except Exception as e:
        print(f"[✗] Ошибка сохранения DICOM SR: {e}")
        return False


# Пример запуска
if __name__ == "__main__":
    json_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_json\post_json\test_atelectasis_2.json"
    dicom_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\pre_dicom\test_atelectasis_2.dcm"
    output_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\post_dicom\test_atelectasis_2_sr.dcm"

    generate_dicom_sr_from_json(json_path, dicom_path, output_path)
