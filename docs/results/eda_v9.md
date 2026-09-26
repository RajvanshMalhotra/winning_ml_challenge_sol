# v9 vs v8: what changed and what is left (OOF, 150k S1)

- macro F0.5 v8 0.9885 → v9 0.9895; S1s improved 2,227, worse 601, S1s still not perfect 13,721

| group | pairs |
|---|---|
| still missed | 12,744 |
| fixed (v8 miss -> v9 found) | 2,007 |
| still wrong | 1,002 |
| new wrong accept | 321 |
| newly lost (v8 found -> v9 miss) | 300 |
| wrong accept removed | 298 |

## Cause x group (pairs)

| cause | fixed (v8 miss -> v9 found) | newly lost (v8 found -> v9 miss) | still missed | new wrong accept | wrong accept removed | still wrong |
|---|---|---|---|---|---|---|
| Indian script name | 40 | 7 | 127 | 19 | 12 | 78 |
| house no. 1-digit typo | 141 | 27 | 457 | 23 | 16 | 174 |
| names share no word | 797 | 121 | 1,314 | 51 | 133 | 113 |
| no address | 846 | 111 | 10,265 | 165 | 96 | 265 |
| other / similar text | 183 | 34 | 581 | 63 | 41 | 372 |

## country x group (share within group)

| country | fixed (v8 miss -> v9 found) | newly lost (v8 found -> v9 miss) | still missed | new wrong accept | wrong accept removed | still wrong |
|---|---|---|---|---|---|---|
| India | 0.434 | 0.447 | 0.359 | 0.461 | 0.483 | 0.450 |
| US | 0.566 | 0.553 | 0.641 | 0.539 | 0.517 | 0.550 |

## same_name_s1s x group (share within group)

| same_name_s1s | fixed (v8 miss -> v9 found) | newly lost (v8 found -> v9 miss) | still missed | new wrong accept | wrong accept removed | still wrong |
|---|---|---|---|---|---|---|
| 0 | 0.750 | 0.703 | 0.460 | 0.558 | 0.742 | 0.535 |
| 1 (unique) | 0.081 | 0.107 | 0.043 | 0.109 | 0.094 | 0.218 |
| 2 | 0.071 | 0.103 | 0.100 | 0.140 | 0.057 | 0.082 |
| 3-5 | 0.053 | 0.040 | 0.142 | 0.106 | 0.047 | 0.058 |
| 6+ | 0.044 | 0.047 | 0.255 | 0.087 | 0.060 | 0.108 |

## zone x group (share within group)

| zone | fixed (v8 miss -> v9 found) | newly lost (v8 found -> v9 miss) | still missed | new wrong accept | wrong accept removed | still wrong |
|---|---|---|---|---|---|---|
| A<0.01 | 0.000 | 0.000 | 0.158 | 0.025 | 0.000 | 0.000 |
| A>0.99 | 0.006 | 0.010 | 0.000 | 0.003 | 0.034 | 0.188 |
| band | 0.994 | 0.990 | 0.842 | 0.972 | 0.966 | 0.812 |

## q9_bin x group (share within group)

| q9_bin | fixed (v8 miss -> v9 found) | newly lost (v8 found -> v9 miss) | still missed | new wrong accept | wrong accept removed | still wrong |
|---|---|---|---|---|---|---|
| 0.05-0.2 | 0.000 | 0.013 | 0.208 | 0.000 | 0.070 | 0.000 |
| 0.2-0.5 | 0.000 | 0.167 | 0.327 | 0.000 | 0.258 | 0.000 |
| 0.5-0.7 | 0.000 | 0.817 | 0.134 | 0.000 | 0.661 | 0.000 |
| <0.05 | 0.000 | 0.003 | 0.331 | 0.000 | 0.010 | 0.000 |
| >=0.7 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |

## Medians per group

| feature | fixed (v8 miss -> v9 found) | newly lost (v8 found -> v9 miss) | still missed | new wrong accept | wrong accept removed | still wrong |
|---|---|---|---|---|---|---|
| p | 0.418 | 0.733 | 0.114 | 0.398 | 0.647 | 0.858 |
| rr | 0.922 | 0.930 | 0.828 | 0.898 | 0.930 | 0.965 |
| q | 0.599 | 0.831 | 0.158 | 0.600 | 0.820 | 0.937 |
| q9 | 0.876 | 0.600 | 0.166 | 0.770 | 0.563 | 0.925 |
| a_share | 0.924 | 0.496 | 0.164 | 0.700 | 0.474 | 1.000 |
| a_cmargin | 0.321 | 0.183 | -0.048 | 0.158 | 0.098 | 0.685 |
| a_crank | 1.000 | 1.000 | 2.000 | 1.000 | 1.000 | 1.000 |
| a_nhi | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| a_ncomp | 49.000 | 40.500 | 99.000 | 50.000 | 38.000 | 22.000 |
| d_crank | 1.000 | 1.000 | 3.000 | 1.000 | 1.000 | 1.000 |
| d_ncomp | 24.000 | 19.000 | 35.500 | 15.000 | 21.500 | 5.000 |
| s1_name_cnt | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| name_sim | 85.000 | 88.020 | 100.000 | 93.617 | 82.353 | 97.143 |
| s1_nconf | 3.000 | 3.000 | 3.000 | 3.000 | 3.000 | 3.000 |

## Still missed: is this S1 the top choice for the record by Model A (a_crank)?

| a_crank | Indian script name | house no. 1-digit typo | names share no word | no address | other / similar text | All |
|---|---|---|---|---|---|---|
| 1 (top) | 86 | 445 | 442 | 2,461 | 552 | 3,986 |
| 2 | 22 | 6 | 348 | 2,002 | 15 | 2,393 |
| 3-5 | 14 | 3 | 290 | 2,052 | 10 | 2,369 |
| 6+ | 5 | 3 | 234 | 3,750 | 4 | 3,996 |
| All | 127 | 457 | 1,314 | 10,265 | 581 | 12,744 |

- still missed but ranked #1 among all S1s by Model A: 3,986 (median a_share 0.630, median p 0.506, median q9 0.410)
- still missed and another S1 ranks higher: 8,758 — the record looks more like a different business

## Where v9's F0.5 loss sits

| S1 outcome | S1 | F0.5 pts | share |
|---|---|---|---|
| missed + wrong | 122 | 0.000 | 0.025 |
| missed only | 12,002 | 0.008 | 0.745 |
| perfect or blocking only | 136,693 | 0.000 | 0.026 |
| wrong only | 1,183 | 0.002 | 0.204 |

## Examples

### fixed (v8 miss -> v9 found)
- [names share no word] p=0.261 q8=0.323 q9=0.933 a_crank=1 a_share=0.99 same-name S1s=0  
  S1 `Safe Semiconductor International P.C.` | `9614 Greenwich Road, Prince William County, VA`  
  S3 `Nexaria` | `9614-9616 Greenwich Rd, Nokesville, Virginia`
- [names share no word] p=0.102 q8=0.137 q9=0.930 a_crank=1 a_share=0.92 same-name S1s=0  
  S1 `OA Classic Cleaning Service P.C.` | `WI, 4764 C, Town Of Pittsfield`  
  S3 `Fayeveracira` | `C, Town Of Pittsfield, Wisconsin`
- [names share no word] p=0.654 q8=0.509 q9=0.893 a_crank=1 a_share=1.00 same-name S1s=0  
  S1 `Shree Holdings` | `Villa No.42, Namaha Lakewood Villas, Rajendranagar, Hyderabad, Telangana`  
  S3 `Korcirairi` | `Villa No.43, Namaha Lakewood Villas, Rajendranagar, Hyderabad, TG`
- [names share no word] p=0.795 q8=0.366 q9=0.964 a_crank=1 a_share=0.50 same-name S1s=0  
  S1 `Delta Activate Inc` | `5723 Wood Street, Chicago, IL`  
  S2 `Gildzeph` | `5723-B WOOD ST, CHICAGO, IL`
- [Indian script name] p=0.052 q8=0.457 q9=0.882 a_crank=1 a_share=0.98 same-name S1s=0  
  S1 `Innovative Marketing Pvt Ltd` | `2598, Gali No 5, Beadon Pura, New Delhi, Delhi, Fourth Floor, Central Delhi`  
  S3 `इनोवेटिव मार्केटिंग प्रा. लि.` | `Hn 324. Fourth Floor, New Delhi, Central Delhi, DL`

### newly lost (v8 found -> v9 miss)
- [names share no word] p=0.892 q8=0.932 q9=0.577 a_crank=1 a_share=0.25 same-name S1s=0  
  S1 `Womens Health Associates Inc` | `263 Butterfield Road, Elmhurst, IL`  
  S3 `Korjaxgild` | `263B Butterfield Road, Elmhurst CITY, Illinois`
- [no address] p=0.692 q8=0.887 q9=0.686 a_crank=2 a_share=0.43 same-name S1s=1  
  S1 `Career Consultants (India) Private Limited` | `Shop No.53, 1St Floor Hanuman Market, Indira Nagar, Lucknow., Lucknow, Uttar Pradesh`  
  S3 `Career Consultants (India) Private` | ``
- [no address] p=0.593 q8=0.776 q9=0.672 a_crank=2 a_share=0.39 same-name S1s=2  
  S1 `Updike American Urban LLC` | `13517 Glendale Avenue, Unit 1105, Glendale, AZ`  
  S2 `Updike American Urban Llc` | ``
- [names share no word] p=0.899 q8=0.764 q9=0.649 a_crank=1 a_share=0.70 same-name S1s=0  
  S1 `Vision Center Inc` | `2929 Pennsylvania Avenue, Unit 202, Washington, DC`  
  S3 `Korcirayuma` | `2929 Pennsylvania Ave, # 202, Washington, District of Columbia`
- [no address] p=0.601 q8=0.764 q9=0.592 a_crank=2 a_share=0.45 same-name S1s=0  
  S1 `Narang Impex Private Limited` | `45 Chandrika Nagar Colony, Varanasi, Uttar Pradesh`  
  S3 `Narang Impex Limited Services` | ``

### still missed
- [no address] p=0.001 q8=0.002 q9=0.002 a_crank=241 a_share=0.00 same-name S1s=224  
  S1 `Cardiology Care` | `Macks Creek, MO, 1734 W Branch Road`  
  S2 `CARDIOLOGY CARE` | ``
- [no address] p=0.747 q8=0.162 q9=0.226 a_crank=1 a_share=0.48 same-name S1s=9  
  S1 `Alexandre, Jeanne E., Esq.` | `602 Rolling Green Drive, High Point, NC`  
  S2 `Alexandre, Jeanne E., Esq.` | ``
- [no address] p=0.011 q8=0.026 q9=0.037 a_crank=34 a_share=0.01 same-name S1s=53  
  S1 `Beth Community Church` | `1240 Wales Avenue, Birmingham, AL`  
  S3 `Beth Community Church Corp` | ``
- [no address] p=0.002 q8=0.003 q9=0.004 a_crank=57 a_share=0.00 same-name S1s=0  
  S1 `Asset Building Institute` | `Crisfield, 19 Somers Cove, MD`  
  S2 `Asset 8uilding Institute Inc.` | ``
- [no address] p=0.316 q8=0.416 q9=0.333 a_crank=2 a_share=0.22 same-name S1s=0  
  S1 `Hyderabad Services Pvt. Ltd.` | `Plot No 40, H No : 110, Janapriya Towers Huda Complex, Saroor Nagar, Hyderabad, Telangana`  
  S3 `Hyderabad Services Pvt. Enterprises` | ``

### new wrong accept
- [names share no word] p=0.237 q8=0.263 q9=0.867 a_crank=2 a_share=0.34 same-name S1s=0  
  S1 `Parevue Interests` | `2192 Lee Highway, Smyth County, VA`  
  S2 `JAXSYNECT0` | `2292C LEE HIGHWAY, MOUNT SIDNEY, VA`
- [no address] p=0.423 q8=0.365 q9=0.705 a_crank=1 a_share=0.85 same-name S1s=2  
  S1 `Continental Materials Brands LLC` | `617 Owen St NW, Cedar Rapids, IA`  
  S2 `Continental Materials Brands` | ``
- [other / similar text] p=0.722 q8=0.633 q9=0.715 a_crank=1 a_share=1.00 same-name S1s=1  
  S1 `Dynamic Cloud Solutions` | `125 Columbia Court, Unit Suite Bay 2, Chaska, MN`  
  S2 `DYNAMIC CLOUD SOLUTIONS CO` | `330 COLUMBIA CT, CHASKA, MN`
- [no address] p=0.919 q8=0.496 q9=0.795 a_crank=1 a_share=0.69 same-name S1s=2  
  S1 `Elite Gifting Pvt Ltd` | `58, Second Floor, Model Town Northex, Delhi, North West, Delhi`  
  S3 `Smt Elite Gifting Pvt-Ltd` | ``
- [house no. 1-digit typo] p=0.023 q8=0.743 q9=0.760 a_crank=1 a_share=1.00 same-name S1s=0  
  S1 `Embree Medical Center` | `21092 Emerald Isle Drive, Lewes, DE`  
  S2 `Embree Medical` | `#21094 EMERALD ISLE DRIVE, LEWES, DE`

### wrong accept removed
- [other / similar text] p=0.254 q8=0.878 q9=0.637 a_crank=1 a_share=0.89 same-name S1s=0  
  S1 `Indian Investment Private Limited` | `No. 201, 2Nd Floor, Unispace Business Centre, Epip Zone, Whitefield, Bangalore, Bangalore, Karnataka`  
  S3 `Investment Private Limited Services` | `Room No 201, ಕರ್ನಾಟಕ, null`
- [no address] p=0.624 q8=0.763 q9=0.696 a_crank=1 a_share=0.70 same-name S1s=1  
  S1 `Arroyo Pennantpark Inc.` | `10856 Nord Avenue, Bloomington, MN`  
  S3 `Arroyo Pennantpark` | ``
- [names share no word] p=0.720 q8=0.933 q9=0.152 a_crank=2 a_share=0.40 same-name S1s=0  
  S1 `Porter British` | `712 Colegate Drive, Marietta, OH`  
  S2 `Veosyn` | `##812 COLEGATE DR, MARIETTA, OH`
- [names share no word] p=0.988 q8=0.939 q9=0.531 a_crank=1 a_share=0.54 same-name S1s=0  
  S1 `La & Co` | `Karnataka, First Floor, 1/1.Palace Road, Bangalore`  
  S3 `Zephjax` | `Karnataka, Bangalore, Palace Road First Floor, 1/1`
- [other / similar text] p=0.171 q8=0.906 q9=0.692 a_crank=1 a_share=1.00 same-name S1s=1  
  S1 `Dharamgarh India LLP` | `Amulya Seeds Pvt Ltd, Charbahal, Dharamgarh, Kalahandi, Orissa`  
  S3 `Dharamgarh India Pvt Ltd` | `Door No 46/6 Amulya Seeds Pvt Ltd, Charbahal, Dharamgarh, Kalahandi, Orissa`

### still wrong
- [no address] p=0.264 q8=0.754 q9=0.828 a_crank=2 a_share=0.34 same-name S1s=2  
  S1 `Steelworkers Local 586` | `5580 Walter Canady Road, Hope Mills, NC`  
  S2 `Steelworkers Local 586 Corporation` | ``
- [house no. 1-digit typo] p=0.931 q8=0.996 q9=0.996 a_crank=1 a_share=1.00 same-name S1s=1  
  S1 `Harbor Horizon Ameren` | `5853 Ridgecrest Avenue, Fauquier County, VA`  
  S3 `Harbor Horizon Ameren Inc.` | `585 Ridgecrest Ave, Fauquier County, Virginia`
- [other / similar text] p=0.932 q8=0.958 q9=0.946 a_crank=1 a_share=1.00 same-name S1s=2  
  S1 `Swami Solutions Private Limited` | `91 Springboard Business Hub Private Limited B-1 / H-3, Mohan Estate, Mathura Road, Delhi, North East, Delhi`  
  S3 `Swami Solutions  Limited` | `Door No 54 Springboard Business Hub Private Limited B-1 / H-3, Mohan Estate, Mathura Road, North East, Delhi, DL`
- [no address] p=0.100 q8=0.794 q9=0.777 a_crank=1 a_share=0.72 same-name S1s=0  
  S1 `Delhi India Clinic` | `101, Prograsive 1St Floor Rohit Kunj Market Pitampura, Delhi, North West, Delhi`  
  S2 `DELHl IIFNA-CLINIC` | ``
- [Indian script name] p=0.256 q8=0.837 q9=0.780 a_crank=1 a_share=0.92 same-name S1s=0  
  S1 `Innovative Global Private Limited` | `East Delhi, Delhi, 245 Agcr Enclave, New Delhi`  
  S2 `इनोवेटिव ग्लोबल प्राइवेट लिमिटेड` | `245/32, EAST DELHI, Delhi`
