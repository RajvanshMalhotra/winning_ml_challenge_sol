from ber.lexicons import load_lexicon
from ber.text import (domain_body, fold, is_domain_name, normalize_address, normalize_name,
                      record_text, segment)

LEX = load_lexicon()


def test_fold():
    assert fold("Tri-State Córnerstone Bérto") == "tri-state cornerstone berto"


def test_name_suffix_and_core():
    assert normalize_name("Tri-State Cornerstone Berto  [LLC]", LEX) == ("tri state cornerstone berto llc", "tri state cornerstone berto", "llc")
    assert normalize_name("Galaxy Solutions Pvt Ltd.", LEX)[1:] == ("galaxy solutions", "pvt_ltd")
    assert normalize_name("Solutions Jain Benefit of Noida Private", LEX)[1:] == ("solutions jain benefit noida", "pvt_ltd")
    assert normalize_name("Europ & Frères Distribution S.A.", LEX)[1:] == ("europ freres distribution", "sa")
    assert normalize_name("ZNB Club SARL", LEX)[2] == "sarl"


def test_name_fillers_dedupe_and_reorder():
    norm, core, suf = normalize_name("The The Morgan, Leonanie F., O.D., DDS PC", LEX)
    assert norm == "morgan leonanie f o d dds pc" and suf == "pc"
    assert normalize_name("M/s #southerneducational", LEX)[0] == "southerneducational"
    assert normalize_name("Co Nautical Center", LEX)[1:] == ("nautical center", "co")
    assert normalize_name("Nautical & Co", LEX)[1:] == ("nautical", "co")


def test_name_core_never_empty():
    assert normalize_name("Pvt Ltd", LEX)[1] == "pvt ltd"


def test_address_us():
    assert normalize_address("1500 JUPITER RD, PO BOX 8832, ALLEN, TX", "US", LEX) == (
        "1500 jupiter road po box 8832 allen texas", "1500", "1500 8832")
    assert normalize_address("Texas, # 609, Allen, 1500 Jupiter Road", "US", LEX)[1] == "1500"
    assert normalize_address("01130 REGENCY ROAD, ATL, GA", "US", LEX)[1] == "1130"
    assert normalize_address("1130- Regency Road, Atlanat, Georgia", "US", LEX)[1] == "1130"


def test_address_india():
    norm, house, nums = normalize_address("NULL, MH, 4-7/1 To 14 Plot No. 15 Airport Road, NULL, Kolhapur", "India", LEX)
    assert house == "47/1" and "maharashtra" in norm and "null" not in norm.split()
    assert normalize_address("DOOR NO 164, MOC SINGAPORE PLAZA", "India", LEX)[:2] == ("164 moc singapore plaza", "164")
    norm = normalize_address("Door No 236 Floor Salarapuria, Bengaluru, ಕರ್ನಾಟಕ, Bangalore", "India", LEX)[0]
    assert norm.split().count("bengaluru") == 2 and "karnataka" in norm


def test_state_map_is_per_country():
    # "GA" is Georgia in the US and Goa in India; unknown countries get no state expansion.
    assert normalize_address("X, GA", "US", LEX)[0] == "x georgia"
    assert normalize_address("X, GA", "India", LEX)[0] == "x goa"
    assert normalize_address("X, GA", "France", LEX)[0] == "x ga"


def test_address_france():
    assert normalize_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France", LEX)[:2] == (
        "63 rue de dieppe lille hauts de france", "63")


def test_domain_detection_and_segmentation():
    assert is_domain_name("fafloonpetcare.com") and is_domain_name("M/s #southerneducational") is False
    assert is_domain_name("#southerneducational") and not is_domain_name("Fafloon Pet Care Inc")
    assert domain_body("www.2827art.com") == "2827art"
    assert segment("fafloonpetcare", {"fafloon", "pet", "care"}) == ["fafloon", "pet", "care"]
    assert segment("2827art", {"art"}) == ["2827", "art"]


def test_record_text():
    assert record_text("A Co", "1 Main St", "US") == "name: A Co | address: 1 Main St | country: US"
