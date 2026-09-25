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
    assert normalize_address("#4332 BASSETT CREEK DRIVE, GOLDEN VALLEY, MN", "US", LEX)[1] == "4332"
    assert normalize_address("#194, 8TH BLOCK, NEW 54, 4TH FLOOR", "India", LEX)[1] == "194"
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


# ---- extra fields (tracker N11-N18) ----
from ber.text import (address_extras, extract_units, fix_mojibake, phonetic_key, skeleton_key,
                      sorted_name, trade_name_parts)


def test_mojibake_removed_but_real_circumflex_kept():
    assert fix_mojibake("Sector Â 15, BLOCK ÂA") == "Sector 15, BLOCK A"
    assert fix_mojibake("Bangalore Â") == "Bangalore"
    assert fix_mojibake("CHÂTEAU ROUGE") == "CHÂTEAU ROUGE"


def test_trade_name_parts():
    assert trade_name_parts("Acme Holdings LLC dba Sunrise Bakery", LEX) == "acme holdings | sunrise bakery"
    assert trade_name_parts("Grain & Fils (Boulangerie Dupont)", LEX) == "grain fils | boulangerie dupont"
    assert trade_name_parts("Morgan Dental (PC)", LEX) == ""  # a bracketed legal suffix is not a trade name
    assert trade_name_parts("Gmax Automobiles (India) Private  Limited", LEX) == ""  # a bracketed country is a qualifier
    assert trade_name_parts("Galaxy Solutions Pvt Ltd", LEX) == ""


def test_units():
    assert extract_units("1500 jupiter rd, po box 8832, unit 609, allen") == "pobox:8832 unit:609"
    assert extract_units("4th floor salarapuria, # 12") == "floor:4 unit:12"
    assert extract_units("1644 crownsville road, fl 0") == "floor:0"
    assert extract_units("12 main street") == ""
    assert extract_units("#4332 bassett creek drive") == ""  # leading '#' is the house number


def test_address_extras_landmark_postal():
    landmark, core, postal, unit = address_extras("12 MG Road, Near SBI ATM, Pune, 411001", "India", LEX)
    assert landmark == "near sbi atm" and "sbi" not in core.split() and postal == "411001" and unit == ""
    assert address_extras("Opp. City Mall, 4 Park St, Kolkata 700016", "India", LEX)[2] == "700016"
    assert address_extras("DOOR NO 461 805, A WING, JOGESHWARI WEST", "India", LEX)[2] == ""  # door number, not a PIN
    assert address_extras("1500 Jupiter Road, Allen, TX, 75002", "US", LEX)[2] == "75002"
    assert address_extras("12345 Main Street, Austin, TX", "US", LEX)[2] == ""  # 5-digit house number, not a ZIP
    assert address_extras("6901 110, Round ROCK, # 11108, Texas", "US", LEX)[2] == ""  # '#' marks a unit
    assert address_extras("20 Rue Parmentier, 59140 Dunkerque", "France", LEX)[2] == "59140"
    assert address_extras("5 Some Road, 12345", "Atlantis", LEX)[2] == "12345"  # unseen country: generic fallback


def test_name_keys():
    assert sorted_name("nautical center") == "center nautical"
    assert phonetic_key("galaxy solutions") == phonetic_key("galaxi solutions")
    assert skeleton_key("shrinivas") == skeleton_key("srinivas")
    assert skeleton_key("lakshmi") == skeleton_key("laxmi")
    assert skeleton_key("sharma") == skeleton_key("shurma")


def test_extract_state():
    from ber.text import extract_state
    assert extract_state("1500 Jupiter Rd, Allen, TX", "US", LEX) == "texas"
    assert extract_state("Texas, # 609, Allen", "US", LEX) == "texas"
    assert extract_state("Door No 236, Bengaluru, ಕರ್ನಾಟಕ", "India", LEX) == "karnataka"
    assert extract_state("NULL, MH, 47/1 Airport Road", "India", LEX) == "maharashtra"
    assert extract_state("12 Main Street, Springfield", "US", LEX) == ""
    assert extract_state("1600 Penn Ave, Washington, DC", "US", LEX) == "district of columbia"  # city named Washington
    assert extract_state("45 Main St, Washington, Utah", "US", LEX) == "utah"
    assert extract_state("Washington, 12 Pine St, Seattle, WA", "US", LEX) == "washington"
    assert extract_state("20 Rue Parmentier, Dunkerque, Nord", "France", LEX) == ""  # no state lexicon
