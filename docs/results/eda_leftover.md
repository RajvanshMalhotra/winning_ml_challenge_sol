# What is left after each stage (OOF, B split, 150k S1)

- S1 businesses: 150,000 (8,496 singletons); candidate pairs: 19,476,117
- True pairs: 518,414; in the shortlist: 517,895; **lost at blocking: 519**

| stage | tau | macro_F0.5 | found | missed (shortlist) |   of which above tau but lost 1-owner | wrong accepts | S1 not perfect | singletons wrong |
|---|---|---|---|---|---|---|---|---|
| A | 0.7000 | 0.9779 | 496,386 | 21,509 | 9 | 4,496 | 23,077 | 256 |
| v5 | 0.6500 | 0.9834 | 501,551 | 16,344 | 15 | 3,247 | 18,121 | 186 |
| v8 | 0.7500 | 0.9885 | 503,144 | 14,751 | 1 | 1,300 | 15,213 | 80 |

## How pairs move through the stages (A / v5 / v8; Y = accepted)

| A/v5/v8 | true pairs | wrong pairs (accepted at least once) |
|---|---|---|
| YYY | 493,848 | 746 |
| nnn | 12,216 | 0 |
| nYY | 5,644 | 187 |
| nnY | 3,166 | 336 |
| YYn | 1,576 | 1,470 |
| YnY | 486 | 31 |
| nYn | 483 | 844 |
| Ynn | 476 | 2,249 |

Read: `nnY` = missed by A and v5, fixed by v8; `YYn` = found by A and v5, lost by v8.

# What v8 still gets wrong

Missed true pairs in the shortlist: **14,751**; wrong accepts: **1,300**; lost at blocking: **519**

## Missed true pairs, by main cause (first matching rule wins)

| cause | pairs | share |
|---|---|---|
| no address | 11,111 | 0.7530 |
| names share no word | 2,110 | 0.1430 |
| similar text, still rejected | 764 | 0.0520 |
| house no. 1-digit typo | 598 | 0.0410 |
| Indian script name | 167 | 0.0110 |
| lost 1-owner to another S1 | 1 | 0.0000 |

## Missed pairs: cause x Model A zone

| cause | A<0.01 (not reranked) | A>0.99 | band (reranked) | All |
|---|---|---|---|---|
| Indian script name | 60 | 2 | 105 | 167 |
| house no. 1-digit typo | 47 | 0 | 551 | 598 |
| lost 1-owner to another S1 | 0 | 0 | 1 | 1 |
| names share no word | 223 | 9 | 1,878 | 2,110 |
| no address | 1,574 | 0 | 9,537 | 11,111 |
| similar text, still rejected | 107 | 5 | 652 | 764 |
| All | 2,011 | 16 | 12,724 | 14,751 |

## Missed vs found (sampled 100k found), side by side

**country**

| country | missed | found |
|---|---|---|
| US | 0.6310 | 0.5990 |
| India | 0.3690 | 0.4010 |

**src**

| src | missed | found |
|---|---|---|
| S3 | 0.5200 | 0.5170 |
| S2 | 0.4800 | 0.4830 |

**cand_noaddr**

| cand_noaddr | missed | found |
|---|---|---|
| 1 | 0.7530 | 0.0220 |
| 0 | 0.2470 | 0.9780 |

**native_script**

| native_script | missed | found |
|---|---|---|
| 0 | 0.9860 | 0.9260 |
| 1 | 0.0140 | 0.0740 |

**house**

| house | missed | found |
|---|---|---|
| missing on one side | 0.8160 | 0.2420 |
| same | 0.0690 | 0.6250 |
| 1-digit typo | 0.0600 | 0.0700 |
| different | 0.0550 | 0.0640 |

**postal_same**

| postal_same | missed | found |
|---|---|---|
| missing | 1.0000 | 1.0000 |
| same | 0.0000 | 0.0000 |

**cluster_size**

| cluster_size | missed | found |
|---|---|---|
| 4-6 | 0.5990 | 0.5920 |
| 2-3 | 0.2960 | 0.3080 |
| 7+ | 0.0870 | 0.0840 |
| 1 | 0.0180 | 0.0160 |

**zone**

| zone | missed | found |
|---|---|---|
| band (reranked) | 0.8630 | 0.1220 |
| A<0.01 (not reranked) | 0.1360 | 0.0000 |
| A>0.99 | 0.0010 | 0.8780 |

**Medians**

| feature | missed | found | wrong accept |
|---|---|---|---|
| p | 0.1493 | 0.9997 | 0.8064 |
| rr | 0.8516 | 1.0000 | 0.9453 |
| q | 0.2146 | 0.9992 | 0.9007 |
| name_sim | 100.0000 | 100.0000 | 93.4409 |
| addr_sim | 0.0000 | 100.0000 | 95.1669 |
| name_shared_words | 2.0000 | 2.0000 | 2.0000 |
| s1_name_cnt | 1.0000 | 1.0000 | 0.0000 |
| d_crank | 2.0000 | 1.0000 | 1.0000 |
| d_ncomp | 33.0000 | 3.0000 | 7.0000 |
| n_cands | 106.0000 | 102.0000 | 104.0000 |
| p_rank | 5.0000 | 2.0000 | 4.0000 |

## Missed pairs: final score q

| q bin | pairs | share |
|---|---|---|
| (-0.001, 0.05] | 4,308 | 0.2920 |
| (0.05, 0.2] | 2,842 | 0.1930 |
| (0.2, 0.5] | 4,594 | 0.3110 |
| (0.5, 0.75] | 3,006 | 0.2040 |
| (0.75, 1.0] | 1 | 0.0000 |

- Missed pairs whose S1 found **none** of its matches: 377; S1 found some: 14,374
- Of 1 lost to one-owner: the winning S1 has the SAME core name in 0.0%

## Wrong accepts

| cause | another S1 | no S1 (unmatched record) | All |
|---|---|---|---|
| Indian script name | 3 | 19 | 22 |
| house no. 1-digit off | 4 | 186 | 190 |
| house no. different | 5 | 106 | 111 |
| names share no word | 193 | 121 | 314 |
| no address | 304 | 57 | 361 |
| same/similar name + address | 26 | 276 | 302 |
| All | 535 | 765 | 1,300 |

- On singleton S1s: 84 wrong accepts; by zone: band (reranked) 1,102, A>0.99 198

## Where the F0.5 loss sits (per S1)

| S1 outcome | S1 | share_of_loss | F0.5 pts |
|---|---|---|---|
| blocking only | 414 | 0.0240 | 0.0003 |
| missed + wrong | 130 | 0.0240 | 0.0003 |
| missed only | 13,519 | 0.7720 | 0.0089 |
| perfect | 134,787 | 0.0000 | 0.0000 |
| wrong only | 1,150 | 0.1800 | 0.0021 |

| country | S1 | loss |
|---|---|---|
| India | 60,031 | 0.0112 |
| US | 89,969 | 0.0116 |

## Examples

### missed: no address
- p=0.012 rr=0.391 q=0.042  
  S1 `Alankar Group` | `Bachan Singh Marg Vpo Ladhowal Ludhiana, Ludhiana, Punjab`  
  S3 `Shri Alankar Co` | ``
- p=0.122 rr=0.969 q=0.289  
  S1 `Mumbai Sons Ltd` | `A 301, Krishna Galaxy Apartment, Dutta Mandhir Road, Vakola, Santacruz(E), Mumbai, Mumbai City, Maharashtra`  
  S2 `Mumbai Sons` | ``
- p=0.003 rr=nan q=0.008  
  S1 `Martinez & Gallagher Seas Inc` | `430 230, Monroe, UT`  
  S2 `Martinez & (Gallagher)` | ``
- p=0.422 rr=0.969 q=0.645  
  S1 `XA Tax Private Limited` | `No 9/1 Indira Nagar, 2Nd Street Odakkadu, Tiruppur, Coimbatore, Tamil Nadu`  
  S2 `XA Private Tax Limited (ID: 68674)` | ``
- p=0.155 rr=0.816 q=0.496  
  S1 `Supreme Biomedical Concepts` | `4702 Memorie Lane, Klamath Falls, OR`  
  S2 `Supreme (Contehots) Biomedical` | ``
- p=0.008 rr=nan q=0.009  
  S1 `Internal Medicine Specialists` | `NY, Greenburgh, 64 Mclean Avenue`  
  S3 `Internal Medicie Specialists` | ``

### missed: names share no word
- p=0.742 rr=0.918 q=0.637  
  S1 `Wyche, Matthews & Pulliam Education, Inc.` | `214 Laurel Lane, Nicholasville, KY`  
  S3 `Avinovi` | `KY, Nicholasville, 214 Laurel Lane`
- p=0.924 rr=0.945 q=0.592  
  S1 `Dermatology Specialists of Syracuse` | `300 Audubon Parkway, Unit Unit 33, Syracuse, NY`  
  S3 `Solbrixkor` | `300 Audubon Parkway, Unit Unit 33, Syracuse, New York`
- p=0.069 rr=0.930 q=0.701  
  S1 `Hotel Enterprises Private Limited` | `22, N.S. Road Po - G.P.O, Kolkata, Kolkata, Howrah, West Bengal`  
  S2 `Jaxsyn` | `2-2, N.S. ROAD PO - G.P.O, KOLKATA, HOWRAH, KOLKATA, West Bengal`
- p=0.231 rr=0.926 q=0.426  
  S1 `Ventures Integrated India Private Limited` | `Mumbai, Fort, B-16/18, Rpi House (Vatsa Hous) Janmbhoomi Marg, Maharashtra`  
  S2 `EVOVANTAGE` | `B-16/18, FORT, MUMBAI, Maharashtra`
- p=0.131 rr=0.934 q=0.214  
  S1 `Uptown Tattoo!` | `50 Allison Drive, Cherokee, NC`  
  S2 `Kelobelo` | `50 ALLISON DRIVE, CHEROEKE, NC`
- p=0.250 rr=1.000 q=0.553  
  S1 `Ventures Rmi Estates Group` | `Bangalore, No 5, Bangalore, Karnataka, Ramakrishna Street, Sheshadripuram Sheshadripuram`  
  S2 `rmiestates.com` | `BENGALURU, NO 9-5, BANGALORE, Karnataka`

### missed: similar text, still rejected
- p=0.982 rr=0.148 q=0.621  
  S1 `3378 Lincoln Street Management P.C.` | `1394 Amherst Street, Unit Apartment 14, Buffalo, NY`  
  S3 `3378 Lincoln Street` | `Buffalo CCDP, Unit Apartment 14, New York, 01394 Amherst Street`
- p=0.372 rr=0.992 q=0.677  
  S1 `Gulf Grand Direct L.L.C.` | `2176 Cochran Drive, Gresham, OR`  
  S2 `Gulf Grand L.L.C. Direct` | `2181 COCHRAN DR, GRESHAM, OR`
- p=0.925 rr=0.138 q=0.743  
  S1 `Shanda K. Udell, M.D.` | `IL, Bridgeview, 7837 73rd Street`  
  S2 `Shanda K. Udell, M.D. Ltd` | `7586 73RD STREET, BRIDGEVIEW, IL`
- p=0.242 rr=0.969 q=0.695  
  S1 `Gourav Infotech Limited` | `C/O Shivramka Fabrics Pvt. Ltd., Krishnapur Madanpur, Barasat - I, North 24 Parganas, West Bengal`  
  S2 `GOURAV [INFOTECH]` | `NO 47 C/O SHIVRAMKA FABRICS PVT. LTD., KRISHNAPUR MADANPUR, BARASAT - I, West Bengal`
- p=0.001 rr=nan q=0.001  
  S1 `Seattle Oncology Certified Health Associates` | `526 A Malden Avenue, Seattle, WA`  
  S3 `Seattle Oncology Certified  Health` | `##523 Malden Avenue, Seattle, Washington`
- p=0.851 rr=0.007 q=0.306  
  S1 `Shree Clinic` | `D.No.25-3-714, Nellore, Andhra Pradesh`  
  S2 `Shree Clínic Limited` | `ఆంధ్రప్రదేశ్, DOOR NO 25-3-71 , NELLORE`

### missed: house no. 1-digit typo
- p=0.793 rr=0.024 q=0.283  
  S1 `Woolwine, Malone and Miller Sea Group` | `15521 Earlport Circle, Dallas, TX`  
  S3 `Woolwine, Malone and Miller Sea LLC` | `1552 Earlport Cir, Texas, Dallas`
- p=0.625 rr=0.003 q=0.112  
  S1 `Baptist Association LLC` | `9606 Us 23, Betsy Layne, KY`  
  S2 `Baptist Association` | `9604 Us 23, STANVILLE, KY`
- p=0.979 rr=0.027 q=0.740  
  S1 `American League Inc.` | `15 Lake Mattawa Road, Orange, MA`  
  S3 `inc. american league` | `17 Lake Mattawa Road, Orange, Massachusetts`
- p=0.935 rr=0.314 q=0.612  
  S1 `Juliette Alaniz, Ph.D.` | `417 Moraine Court, Asheville, NC`  
  S3 `Juliette Alaniz, Ph.D. Co` | `41 Moraine Ct, Asheville, North Carolina`
- p=0.986 rr=0.019 q=0.548  
  S1 `Smith, Johnson & Cummings Inc.` | `2850 Crittenden Dr, Unit 416, Louisville, KY`  
  S2 `Smith, Johnson & Cummings Inc` | `2852 CRITTENDEN DR, LOUISVILLE, KY`
- p=0.936 rr=0.007 q=0.308  
  S1 `Julian Imperial` | `1465 Maddox Drive, Conway, AR`  
  S2 `Julian Imperial Ltd` | `1467 MADDOX DRIVE, CONWAY, AR`

### missed: Indian script name
- p=0.084 rr=0.124 q=0.143  
  S1 `Arihant Fortune Engineering` | `Plot No. 11, Gajanan Nagar, Nagpur, Maharashtra`  
  S2 `अरिहंत फॉर्च्यून इंजीनियरिंग` | `BLOCK G-341 1, NAGPUR, महाराष्ट्र`
- p=0.000 rr=nan q=0.003  
  S1 `Shyam Consulting Private Limited` | `H.No.2-2-1123/1/A, New Nallakunta, Musheerabad, Hyderabad, Telangana`  
  S2 `శ్యామ్ కన్సల్టింగ్ ప్రైవేట్ లిమిటెడ్` | `2211243, MUSHEERABAD, HYDERABAD, Andhra Pradesh`
- p=0.341 rr=0.969 q=0.643  
  S1 `Best Logistics Pvt Ltd` | `No.201, 2Nd Floor, Shiva Apartments, 1St Main Road, Nagarbhavi, Bangalore, Karnataka`  
  S2 `ಬೆಸ್ಟ್ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `01, BENGALURU, BANGALORE, ಕರ್ನಾಟಕ`
- p=0.007 rr=nan q=0.029  
  S1 `Future Properties Pvt Ltd` | `H No 377, Kenaya Park, Uchagaon, Karveer, Kolhapur, Maharashtra`  
  S2 `फ्यूचर प्रॉपर्टीज प्रा. लि.` | `KARVEER, KOLHAPUR, DOOR NO 249 654, Maharashtra`
- p=0.343 rr=1.000 q=0.538  
  S1 `United Impex Private Limited` | `No, 19, Ground Floor Kumara Krupa Road, Bangalore, Karnataka`  
  S2 `ಯುನೈಟೆಡ್ ಇಂಪೆಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `NO, BANGALORE, Karnataka`
- p=0.005 rr=nan q=0.017  
  S1 `Universal Care Private Limited` | `No. 28, Batra Centre, Sardar Patel Road, Guindy, Chennai, Tamil Nadu`  
  S3 `யுனிவர்சல் கேர் பிரைவேட் லிமிடெட்` | `22, Batra Centre, Sardar Patel Road, Chennai City Region, Chennai, TN`

### missed: lost 1-owner to another S1
- p=0.973 rr=0.922 q=0.800  
  S1 `W 9 M Orange Inc` | `5420 Carters Chapel Road, TN, Lenoir City`  
  S3 `Fluxsyniri` | `5420 Carters Chapel Road, Lenoir City, Tennessee`
  won by S1 `Jonsson, Preciado & Dyson LLC` | `5420 Carters Chapel Road, Lenoir City, TN`

### wrong accept: no address
- p=0.255 rr=0.945 q=0.795  
  S1 `Atlanta Textile Private Limited` | `A-253, 2Nd Floor, Ishwar Singh Complex, New Delhi, South Delhi, Delhi`  
  S2 `Atlanta Tdrde Private Limited` | ``
- p=0.505 rr=0.891 q=0.849  
  S1 `Rajasthan Fortune (India) Ltd` | `3202-B Ram Bazar Mori Gate, Delhi, Central Delhi, Delhi`  
  S3 `Rajasthan Fortune (India) Ltd.` | ``
- p=0.372 rr=0.984 q=0.755  
  S1 `Good Energy Pvt Ltd` | `Churu, Poonia Colony, Rajasthan`  
  S3 `Good Eaerhmgy Pvt Ltd` | ``
- p=0.959 rr=0.820 q=0.995  
  S1 `Rivera & Vasquez Joint` | `205 Sycamore Street, Snow Shoe, PA`  
  S3 `Rivera & Vasquez Joint LLC` | ``

### wrong accept: names share no word
- p=0.514 rr=0.930 q=0.849  
  S1 `Global Global Private Limited` | `103-B, Shelton Sapphire, Sec-15 Plot No-18-19, Cbd Belapur, Navi Mumbai, Thane, Maharashtra`  
  S3 `Quonyla` | `103-B, Shelton Sapphire, Sec-15, Plot No.18-19, Cbd Belapur, Navi Mumbai, Thane, Maharashtra`
- p=0.421 rr=0.934 q=0.777  
  S1 `First Industries Private Limited` | `147 Mahatma Gandhi Road P S Jorashanko, Calcutta, West Bengal`  
  S3 `Deltabelo` | `#1-47 Mahatma Gandhi Road P S Jorasanko, Kolkata, Calcutta, পশ্চিমবঙ্গ`
- p=0.406 rr=0.918 q=0.769  
  S1 `Beacon P.C.` | `8 Split Rock Road, Pittsford, NY`  
  S3 `Evozeta` | `8 Split Rock Road, Pittsford, New York`
- p=0.503 rr=0.930 q=0.829  
  S1 `Ciavarella Holdings` | `392 Dushane Drive, Tonawanda, NY`  
  S2 `GILDORBIPYRA` | `DUSHANE DRIVE, TONAWANDA, NY`

### wrong accept: same/similar name + address
- p=0.374 rr=1.000 q=0.888  
  S1 `Bangalore Pharmaceuticals Limited` | `63, Muthanallur Pearl City, Bangalore, Karnataka`  
  S3 `Bangalore Ltd Center` | `Door No 63, Bangalore, Banglore, KA`
- p=0.996 rr=nan q=0.995  
  S1 `Elysha Boothe, CPA, DDS PC` | `1895 Melrose Street, Unit 144, AZ, Gilbert`  
  S2 `Elysha Boothe, Cpy, Dds Pc` | `1895 MELROSE ST, GILBERT, AZ`
- p=0.995 rr=nan q=0.996  
  S1 `Zao LLC` | `Unit Apartment 1, NY, 884 212, Saugerties`  
  S3 `Zao Ltd` | `884 212, # Apartment 1, Saugerties, New York`
- p=0.180 rr=0.996 q=0.859  
  S1 `Noida Softech Private Limited` | `D-128, Sector-7, Ghaziabad, Noida, Gautam Buddha Nagar, Uttar Pradesh`  
  S3 `Noida Private Limited Service` | `A-00127-128, Gautam Buddha Nagar, N/A, UP`

### wrong accept: house no. 1-digit off
- p=0.446 rr=0.996 q=0.854  
  S1 `Taylor & Duan` | `3245 135, Stoneville, NC`  
  S3 `Taylor &  Duan LLC #4453` | `3247 135, Stneville, North Carolina`
- p=0.973 rr=1.000 q=0.998  
  S1 `National Dental Associates` | `200 Side Street, Goreville, IL`  
  S3 `National Dental Associates Ltd` | `20 Side St, Goreville, Illinois`
- p=0.822 rr=0.992 q=0.804  
  S1 `Veor Opportunities` | `10235 31st Street, Unit 3, Phoenix, AZ`  
  S2 `Veor Opportunities Corp` | `1023 31ST STREET, CPU KARENS HALLMARK CACTUS, AZ`
- p=0.438 rr=0.996 q=0.950  
  S1 `Noble College` | `#736, Gupta House, 7Th Cross, 3Rd Block, Kormangala, Near Bda Complex, Bangalore, Karnataka`  
  S3 `noble centre` | `#738, Gupta House, 7Th Cross, 3Rd Block, Kormangala, Near Bda Complex, Bangalore, KA`
