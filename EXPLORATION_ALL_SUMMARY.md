# Exploration 绯诲垪瀹為獙鎬荤粨

鐢熸垚鏃ユ湡锛?026-08-19

## 1. 姹囨€昏寖鍥翠笌鍒ゅ畾鍙ｅ緞

鏈姤鍛婅鐩?`exploration`銆乣explorationv2`銆乣explorationv3`銆乣explorationv4` 鍏?470 椤归厤缃€傞€愰」閰嶇疆銆佸疄楠岀洰鐨勩€丳SNR 杞ㄨ抗銆侀/宄?鏈帰閽堛€佽缁冪姸鎬併€佸紓甯稿師鍥犲拰鏁版嵁鏉ユ簮瑙?`EXPLORATION_ALL_RESULTS.csv`銆?
| 鐗堟湰 | 閰嶇疆鏁?| 閲嶅缓椤?| 绠＄悊鍣ㄩ璁粌椤?| 鎴愬姛 | 澶辫触 | 閲嶅缓椤归渶鍏虫敞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exploration | 126 | 111 | 15 | 123 | 3 | 33 |
| explorationv2 | 53 | 38 | 15 | 53 | 0 | 6 |
| explorationv3 | 210 | 185 | 25 | 210 | 0 | 57 |
| explorationv4 | 81 | 81 | 0 | 81 | 0 | 45 |

缁熶竴鐨勨€滈渶鍏虫敞鈥濆垽瀹氫负锛氭渶缁?PSNR 姣斿嘲鍊煎洖钀借秴杩?1 dB锛屾垨鏈€缁?PSNR 鐩稿棣栦釜鎺㈤拡鎻愬崌涓嶈冻 0.1 dB锛涜缁冨け璐ユ垨鎺㈤拡缂哄け涔熻鍏ラ渶鍏虫敞銆倂3 鐩存帴閲囩敤瀹樻柟姹囨€荤殑涓ユ牸鏍￠獙缁撴灉锛泇1/v2 鏄寜鍚屼竴闃堝€煎洖婧绠楋紱v4 鍥犲皻鏈敓鎴愭眹鎬?TSV锛屼緷鎹?81 浠藉畬鏁存棩蹇楅噸寤恒€?
NeuralExpert 鐨?manager-pretraining 閰嶇疆涓嶄骇鐢熼噸寤?PSNR锛屽洜姝や笉璁″叆鈥滈噸寤洪」闇€鍏虫敞鈥濄€傚師濮?v3 `needs_attention.txt` 浼氬洜缂烘帰閽堝垪鍑鸿繖 25 椤癸紝鏈姤鍛婂皢瀹冧滑褰掍负鈥滄垚鍔熷畬鎴愩€侀噸寤烘寚鏍囦笉閫傜敤鈥濄€?
琛ㄤ腑鐨勨€滃钩鍧囨渶缁?PSNR鈥濇槸鍚屼竴 profile 瀵规墍瑕嗙洊鐩爣鐨勯潪鍔犳潈绠楁湳骞冲潎銆傚畠閫傚悎鍦ㄥ悓涓€鏂规硶鍐呴儴姣旇緝缁撴瀯鎴?Size锛屼笉搴旀妸鍗曠洰鏍囨柟娉曠殑鍧囧€间笌 MC-INR/VarExpert 鐨勫鍙橀噺 aggregate PSNR 鐩存帴妯悜鎺掑悕銆?
## 2. exploration锛歋ize163 缁撴瀯鎼滅储

### 瀹為獙鐩殑

绗竴杞湪 Size163 棰勭畻鍜岀粺涓€ 50 epoch-equivalent 鎺㈤拡涓嬶紝绯荤粺妫€鏌ョ綉缁滄繁搴︺€佷笓瀹舵暟鍜岀綉鏍?瑙ｇ爜鍣ㄩ绠楀垎閰嶇瓑缁撴瀯鍙橀噺銆備富瑕佺洰鏍囦笉鏄骇鐢熸寮?RD 鏇茬嚎锛岃€屾槸蹇€熸壘鍑烘瘡涓柟娉曟棌鐨勫彲鐢ㄧ粨鏋勪笌鏄庢樉鏁呴殰銆?
### 鍚勫疄楠岀粍缁撴灉

| 鏂规硶鏃?| 姣旇緝鍙橀噺 | 鏈€浼?profile | 骞冲潎鏈€缁?PSNR (dB) | 缁撹 |
| --- | --- | --- | ---: | --- |
| SIREN | depth2 / depth3 / depth5 | depth3 | 37.829 | depth3 鐣ヤ紭浜?depth5锛?7.669锛夊拰 depth2锛?6.922锛夛紝涓夎€呭潎绋冲畾銆?|
| CoordNet | res5 / res10 / res15 | res5 | 39.821 | res5 鏄庢樉鏈€浣充笖 5/5 绋冲畾锛況es10銆乺es15 鍒嗗埆鏈?4/5銆?/5 椤瑰洖钀芥垨鏃犲鐩婏紝娣卞害澧炲姞鍙嶈€屾伓鍖栥€?|
| MoE-INR | 4 / 7 / 10 experts | experts10 | 37.220 | 涓撳鏁板鍔犲甫鏉ュ皬骞呭潎鍊兼敹鐩婏紱experts10 鏃犲紓甯革紝experts4/7 鍚勬湁 1 椤归渶鍏虫敞銆?|
| VarExpert | 4 / 6 / 8 experts锛屽浐瀹?top-3 | experts8 | 43.936 | experts8 鐣ラ珮浜?experts4锛?3.802锛夛紝experts6 杈冧綆锛?2.465锛夛紱涓撳鏁版敹鐩婂苟闈炲崟璋冦€?|
| MC-INR | depth3_4 / depth5_6 / depth7_8 | 鏃?| 鈥?| 涓夐」鍏ㄩ儴澶辫触锛屾棤娉曞垽鏂粨鏋勪紭鍔ｏ紱闅忓悗鍦?v2 淇鐩爣甯冨眬鍚庨噸璺戙€?|
| NeuralExpert | depth1 / depth2 / depth3 | depth1 | 37.898 | 涓夌娣卞害閮界ǔ瀹氾紱depth1 鏈€楂橈紝缁х画鍔犳繁娌℃湁鏀剁泭銆?5 涓?manager-pretraining 鍧囨垚鍔熴€?|
| APMGSRN | decoder-heavy / balanced / grid-heavy | balanced | 27.156 | balanced 鏈€濂斤紝grid-heavy 娆′箣锛?5.815锛夛紝decoder-heavy 鏈€浣庯紙23.998锛夈€?|
| fV-SRN | decoder-heavy / balanced / grid-heavy | grid-heavy | 27.295 | grid-heavy 鏄庢樉鏈€濂戒笖绋冲畾锛沚alanced 5/5 闇€鍏虫敞锛宒ecoder-heavy 2/5 闇€鍏虫敞銆?|
| RMDSRN | decoder-heavy / balanced / grid-heavy | balanced | 28.256 | balanced 鍧囧€兼渶楂橈紝浣?4/5 闇€鍏虫敞锛涘彟澶栦袱缁?5/5 闇€鍏虫敞锛屾毚闇插嚭璁粌/璋冨害绋冲畾鎬ч棶棰樸€?|

### v1 缁撹

- 杈冩祬鎴栬緝骞宠　鐨勭粨鏋勬櫘閬嶆洿鍙潬锛欳oordNet res5銆丯euralExpert depth1銆丼IREN depth3 鍒嗗埆鑳滃嚭銆?- VarExpert experts8/top-3 鏄湰杞渶濂界殑澶氬彉閲忛厤缃紝鏈€缁?aggregate PSNR 涓?43.936 dB銆?- MC-INR 鐨勪笁椤瑰け璐ュ睘浜庡疄鐜伴棶棰橈紝鑰屼笉鏄彲闈犵殑妯″瀷鑳藉姏缁撹銆?- RMDSRN 鍜?fV-SRN 瀵硅缁冭繃绋嬫洿鏁忔劅锛屽彧鐪嬫渶缁?PSNR 浼氭帺鐩栦腑閫斿嘲鍊煎洖钀姐€?
## 3. explorationv2锛氫慨澶嶉獙璇佷笌瀹氬悜瓒呭弬鏁版悳绱?
### 瀹為獙鐩殑

v2 淇濈暀 v1锛屼笉瑕嗙洊鏃х粨鏋滐紝骞跺洿缁曚笁涓棶棰樺紑灞曞楠岋細淇 MC-INR 鐩爣甯冨眬锛涙妸 NeuralExpert 鎻愰珮鍒?Size326锛涘洿缁?v1 鐨?VarExpert experts8/top-3 瀵圭収鎵弿 9/10 涓撳鍜屽叏閮?top-k銆?
### 鍚勫疄楠岀粍缁撴灉

| 瀹為獙缁?| 涓昏缁撴灉 | 缁撹 |
| --- | --- | --- |
| MC-INR 淇鍚庨噸璺?| depth3_4=38.166銆乨epth5_6=38.036銆乨epth7_8=31.198 dB | 3/3 鎴愬姛锛岃瘉鏄?v1 鐨勫け璐ヤ富瑕佹潵鑷洰鏍囧竷灞€闂銆俤epth3_4 鏈€缁堝€兼渶楂橈紝浣嗘湁鍥炶惤鏍囪锛沝epth5_6 鐣ヤ綆浣嗘洿绋冲畾锛屾槸鏇寸ǔ濡ョ殑鍊欓€夈€?|
| NeuralExpert Size326 | depth1=39.483銆乨epth2=38.908銆乨epth3=39.265 dB | 15 涓噸寤洪」鍜?15 涓鐞嗗櫒椤瑰叏閮ㄦ垚鍔燂紱depth1 鍐嶆鏈€濂斤紝娴呭眰缁撹璺ㄩ绠椾繚鎸佷竴鑷淬€?|
| VarExpert experts/top-k | 鏈€浣?experts9_top4=44.329 dB锛涘叾娆?experts9_top5=44.286銆乪xperts10_top7=44.261 | 鐩稿 experts8_top3 瀵圭収锛?3.936锛夛紝鏈€浣虫彁鍗囩害 0.392 dB銆倀op-k 涓嶆槸瓒婂ぇ瓒婂ソ锛屼笖鑻ュ共缁勫悎瀛樺湪鏄庢樉鍥炶惤锛屽崟娆?seed 鐨勬渶浼樼偣搴斿湪姝ｅ紡瀹為獙涓楠屻€?|

### v2 缁撹

- v1 鐨?MC-INR 闃诲闂寰楀埌瑙ｅ喅锛屽叏閮ㄩ厤缃潎宸插畬鎴愩€?- VarExpert 鐨勬渶浣崇偣钀藉湪 9 涓撳/top-4锛岃€屼笉鏄渶澶т笓瀹舵暟鎴栨渶绋犲瘑璺敱锛涜矾鐢辩█鐤忓害姣斿崟绾鍔犱笓瀹舵洿閲嶈銆?- NeuralExpert 鐨勬祬灞備紭鍔块噸澶嶅嚭鐜帮紝璇存槑 depth1 鏄悗缁?Size 鐭╅樀鐨勫悎鐞嗗熀绾裤€?
## 4. explorationv3锛氬叏姝ｅ紡 Size 閰嶇疆鐑熼浘娴嬭瘯

### 瀹為獙鐩殑

v3 灏嗘寮?RD-curve 娓呭崟涓殑 210 涓厤缃師鏍峰鍒跺埌闅旂鐩綍锛屽彧鎶婅缁冮暱搴︾粺涓€缂╃煭鍒?50 epoch-equivalent锛屽苟姣?5 涓瓑鏁?epoch 鍋氬浐瀹氭牱鏈帰閽堛€傜洰鏍囨槸瑕嗙洊浜斾釜 Size 妗ｄ綅鍜屼節涓柟娉曟棌锛屾鏌ユ寮忛厤缃槸鍚﹁兘璁粌銆佹槸鍚﹂殢瀹归噺鏀瑰杽锛屼互鍙婃槸鍚﹀彂鐢熷闄枫€?
### 鏂规硶鏃忕粨鏋?
涓嬭〃渚濇鍒楀嚭 Size082 / Size163 / Size326 / Size652 / Size1304 鐨勫钩鍧囨渶缁?PSNR锛涙嫭鍙蜂负璇?Size 涓嬧€滈渶鍏虫敞椤?閲嶅缓椤光€濄€?
| 鏂规硶鏃?| 浜斾釜 Size 鐨勫钩鍧囨渶缁?PSNR (dB) | 鏈€浣?Size | 涓昏鍒ゆ柇 |
| --- | --- | --- | --- |
| APMGSRN | 25.253(0/5), 27.587(0/5), 29.083(0/5), 29.290(0/5), 29.211(0/5) | Size652 | 绋冲畾鎵╁睍鍒?Size652锛孲ize1304 宸插熀鏈ケ鍜屻€?|
| CoordNet | 36.447(1/5), 35.787(4/5), 29.203(4/5), 20.780(5/5), 15.436(5/5) | Size082 | 瀹归噺瓒婂ぇ鍙嶈€岃秺宸紝Size652/1304 鍏ㄩ潰濉岄櫡锛屾槸 v3 鏈€鏄庣‘鐨勭郴缁熸€ф晠闅溿€?|
| fV-SRN | 27.656(1/5), 27.395(1/5), 28.660(2/5), 28.732(2/5), 30.338(2/5) | Size1304 | 鎬讳綋闅忓閲忔敼鍠勶紝浣嗚建杩瑰櫔澹板拰宄板€煎洖钀借緝澶氥€?|
| MC-INR | 38.648(0/1), 38.166(1/1), 39.918(0/1), 39.587(0/1), 38.612(0/1) | Size326 | Size326 鏈€浣筹紝缁х画澧炲ぇ娌℃湁鏀剁泭锛涗粎 Size163 琚爣璁板洖钀姐€?|
| MoE-INR | 35.983(0/5), 36.862(1/5), 37.119(1/5), 38.229(0/5), 37.249(1/5) | Size652 | 鍒?Size652 鍩烘湰姝ｅ悜鎵╁睍锛孲ize1304 鐣ュ洖钀姐€?|
| NeuralExpert | 35.574(0/5), 37.898(0/5), 39.483(0/5), 41.742(0/5), 43.357(0/5) | Size1304 | 浜旀。鍗曡皟鎻愬崌涓?25 涓噸寤洪」鍏ㄩ儴绋冲畾锛屾槸鏈€娓呮櫚鐨勫閲忔墿灞曟洸绾裤€?|
| RMDSRN | 28.699(5/5), 28.700(5/5), 29.418(4/5), 30.209(5/5), 29.700(5/5) | Size652 | 24/25 椤归渶鍏虫敞锛屾棫璋冨害鍦ㄧ煭绋嬭缁冧腑涓ラ噸涓嶅尮閰嶏紱杩欑洿鎺ヨЕ鍙?v4 璋冨害淇銆?|
| SIREN | 36.799(0/5), 37.829(0/5), 38.340(0/5), 38.553(0/5), 35.352(2/5) | Size652 | 鍒?Size652 绋冲畾鎻愬崌锛孲ize1304 鍑虹幇閫€鍖栧拰涓ら」濉岄櫡銆?|
| VarExpert | 41.385(0/1), 42.833(0/1), 43.119(0/1), 42.793(0/1), 42.450(0/1) | Size326 | Size326 杈惧嘲锛屽悗缁閲忔病鏈夎浆鍖栦负 50-epoch 鐭▼鏀剁泭锛屼絾浜旀。鍧囬€氳繃绋冲畾鎬ф鏌ャ€?|

v3 鐨?57 涓噸寤哄紓甯搁」涓紝33 椤瑰悓鏃舵弧瓒斥€滃嘲鍊煎洖钀解€濆拰鈥滄棤鏈夋晥澧炵泭鈥濓紝16 椤逛粎宄板€煎洖钀斤紝8 椤逛粎鏃犳湁鏁堝鐩娿€傛寜鏂规硶鏃忚锛歊MDSRN 24銆丆oordNet 19銆乫V-SRN 8銆丮oE-INR 3銆丼IREN 2銆丮C-INR 1锛汚PMGSRN銆丯euralExpert銆乂arExpert 涓?0銆?
### v3 缁撹

- 鈥滄洿澶фā鍨嬪繀鐒舵洿濂解€濅笉鎴愮珛銆侼euralExpert 鑳界ǔ瀹氭墿灞曪紝CoordNet 鍒欓殢鐫€ Size 澧炲ぇ绯荤粺鎬уけ绋炽€?- Size326鈥揝ize652 宸叉槸澶氫釜鏂规硶鐨勭煭绋嬫敹鐩婄敎鐐癸細MC-INR銆乂arExpert 鍦?Size326 杈惧嘲锛孉PMGSRN銆丮oE-INR銆丼IREN銆丷MDSRN 鍦?Size652 杈惧嘲銆?- v3 鏄煭绋嬬儫闆炬祴璇曪紝涓嶆槸瀹屾暣璁粌鍚庣殑鏈€缁?RD 鎺掑悕銆傚浜庢參鏀舵暃鏂规硶锛孲ize1304 鐨勭煭绋嬪姡鍔垮彲鑳藉寘鍚缁冮绠椾笉瓒筹紱浣嗗ぇ骞呭洖钀戒粛鏄繀椤讳慨澶嶇殑浼樺寲闂銆?
## 5. explorationv4锛欳oordNet 绋冲畾鎬т笌 RMDSRN 璋冨害/鎹熷け娑堣瀺

### 5.1 CoordNet锛氱瓑鍙傛暟娣卞害鎵弿

瀹為獙鍦?Size326銆丼ize652銆丼ize1304 鐨?GT/H_plus/He 涓婏紝鐢ㄦ暣鏁板搴﹀尮閰嶆寮?res10 鍙傛暟閲忥紝姣旇緝 res2/res3/res5/res7/res10锛屽苟淇濈暀姝ｅ紡瀛︿範鐜囥€?
| Size | res2 | res3 | res5 | res7 | res10 | res2 鐩稿 res10 鐨勫尮閰嶄腑浣嶆彁鍗?|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Size326 | 37.748 (0/3) | 36.313 (1/3) | 28.894 (2/3) | 29.496 (2/3) | 23.774 (3/3) | +17.405 dB |
| Size652 | 33.071 (2/3) | 28.742 (2/3) | 21.721 (3/3) | 19.649 (2/3) | 15.106 (3/3) | +22.360 dB |
| Size1304 | 26.090 (2/3) | 21.955 (3/3) | 13.527 (3/3) | 7.467 (3/3) | 7.465 (3/3) | +29.881 dB |

鎷彿浠嶄负鈥滈渶鍏虫敞椤?閲嶅缓椤光€濄€傛祬灞傜粨鏋勫湪涓変釜 Size 涓婇兘鏄捐憲浼樹簬 res10锛屼笖宸窛闅?Size 澧炲ぇ銆傝繖璇存槑 v3 鐨?CoordNet 澶辫触涓嶆槸鍙傛暟閲忎笉瓒筹紝鑰屾槸娣辩綉缁滃湪褰撳墠浼樺寲璁剧疆涓嬬殑绋冲畾鎬ч棶棰樸€?
### 5.2 CoordNet锛歋ize1304 鍥犳灉鎺у埗

| profile | 骞冲潎鏈€缁?PSNR (dB) | 闇€鍏虫敞 | 鐩稿 res10_base_lr 鐨勫尮閰嶄腑浣嶆彁鍗?| 鍒ゆ柇 |
| --- | ---: | ---: | ---: | --- |
| res10_base_lr | 7.465 | 3/3 | 0.000 | 鍩虹嚎鍏ㄩ潰濉岄櫡銆?|
| res10_clip | 7.858 | 3/3 | +1.118 | 鍗曠嫭姊害瑁佸壀鍩烘湰鏃犳晥銆?|
| res10_scaled_lr | 35.584 | 2/3 | +33.126 | 闄嶄綆瀛︿範鐜囧ぇ骞呮仮澶嶆渶缁堣川閲忥紝浣?H_plus/He 浠嶄粠鏇撮珮宄板€煎洖钀姐€?|
| res5_scaled_lr_clip | 36.579 | 0/3 | +32.733 | 涓変釜鐩爣鍏ㄩ儴绋冲畾锛屾槸 v4 鏈€鍙潬鐨?Size1304 鏂规銆?|

鍥犳锛屽涔犵巼鏄富瑕佽嚧鍥狅紝杈冩祬娣卞害鍜屾搴﹁鍓叡鍚屾彁渚涢澶栫ǔ瀹氭€с€傝嫢瑕佹洿鏂版寮忛厤缃紝浼樺厛鍊欓€夋槸 `res5_scaled_lr_clip`锛岃€屼笉鏄彧缁?res10 鍔?clipping銆?
### 5.3 RMDSRN锛?00k 璋冨害淇鍜?lambda 娑堣瀺

RMDSRN 鍙缁?75k 姝ワ紝浣?LR 鍜?lambda 娌挎寮?900k 姝ヨ皟搴︾殑鍓?75k 姝ユ帹杩涖€俙lambda_max=10` 瑕嗙洊浜斾釜 Size锛沗lambda_max=1/0` 浠呭湪 Size082 鍜?Size1304 鍋氬鐓с€?
| Size | lambda=0 | lambda=1 | lambda=10 | 鍒ゆ柇 |
| --- | ---: | ---: | ---: | --- |
| Size082 | 33.796 (0/3) | 33.841 (0/3) | 32.164 (0/3) | lambda=1 鐣ヤ紭锛宭ambda=10 宸叉湁绾?1.68 dB 鍧囧€兼崯澶便€?|
| Size163 | 鈥?| 鈥?| 32.231 (1/3) | lambda=10 鏈?1 椤瑰紓甯搞€?|
| Size326 | 鈥?| 鈥?| 32.763 (1/3) | lambda=10 鏈?1 椤瑰紓甯搞€?|
| Size652 | 鈥?| 鈥?| 34.264 (2/3) | lambda=10 鏈?2 椤瑰紓甯搞€?|
| Size1304 | 37.040 (0/3) | 36.648 (0/3) | 33.764 (2/3) | lambda=0 鏈€浣充笖绋冲畾锛沴ambda=10 姣?lambda=0 浣?3.276 dB銆?|

璋冨害淇鍚庯紝lambda=0/1 鐨勪袱涓鐓?Size 鍧?6/6 绋冲畾锛沴ambda=10 鍦ㄤ簲涓?Size 涓湁 6/15 椤归渶鍏虫敞銆傜粨鏋滆〃鏄庯紝楂樻潈閲嶆柟宸鍒欐樉钁楃壓鐗查噸寤猴紝涓斿奖鍝嶅湪澶фā鍨嬩笂鏇存槑鏄俱€傝嫢鐩爣棣栧厛鏄噸寤?PSNR锛屽簲浼樺厛浣跨敤 lambda=0 鎴?1锛涜嫢蹇呴』淇濈暀涓嶇‘瀹氭€у缓妯★紝闇€瑕侀噸鏂板钩琛?lambda锛岃€屼笉鑳界洿鎺ユ部鐢?10銆?
### v4 鏁版嵁闄愬埗

v4 鐨?81 椤圭姸鎬佸潎涓烘垚鍔燂紝浣嗘湰鍦版壒娆＄洰褰曠己灏戦鏈熺殑 `exploration_summary.tsv` 鍜?`profile_summary.tsv`銆傛湰鎶ュ憡浠庢瘡浠芥棩蹇楃殑鍗佷釜 `Exploration PSNR` 琛屾仮澶嶈建杩癸紝骞舵彁鍙?RMDSRN 鏈€缁?step 鐨?total/member loss銆乿ariance KL銆乴ambda 鍜?LR銆傝璁℃枃妗ｆ彁鍒扮殑 variance-error Pearson correlation 涓?top-1%/top-5% hit rate 娌℃湁鎵撳嵃鍦ㄦ棩蹇椾腑锛屽洜姝ゅ綋鍓?CSV 涓嶅寘鍚繖浜涗笁绫绘寚鏍囷紱濡傞渶鍒嗘瀽涓嶇‘瀹氭€ц川閲忥紝闇€瑕佹仮澶嶈繙绔?`runs/exploration_v4` 鐨?metrics 鏂囦欢鍚庨噸鏂拌繍琛?v4 姹囨€昏剼鏈€?
## 6. 璺ㄧ増鏈患鍚堢粨璁?
1. **缁撴瀯骞堕潪瓒婃繁瓒婂ソ銆?* CoordNet res5/res2銆丯euralExpert depth1銆丼IREN depth3 閮戒紭浜庢洿娣卞€欓€夈€傚挨鍏?CoordNet 鐨勬繁搴︽儵缃氫細闅?Size 鏀惧ぇ銆?2. **瀹归噺鎵╁睍渚濊禆浼樺寲閰嶇疆銆?* NeuralExpert 鍛堢ǔ瀹氬崟璋冩墿灞曪紱CoordNet 鍦ㄥ師瀛︿範鐜囦笅鍛堝弽鍚戞墿灞曘€倂4 璇佹槑闄嶄綆瀛︿範鐜囧彲鎭㈠澶ч儴鍒嗚川閲忥紝娴呭眰鍔?clipping 鎵嶈兘鍚屾椂鎭㈠绋冲畾鎬с€?3. **VarExpert 鍦ㄧ揣鍑戦绠椾笅琛ㄧ幇寮猴紝浣?top-k 鏈€浼樼偣闈炲崟璋冦€?* v2 鐨?experts9/top-4 鏈€濂斤紝鍙瘮 experts8/top-3 楂樼害 0.39 dB锛屽缓璁敤澶?seed 纭宸紓鏄惁绋冲仴銆?4. **RMDSRN 鐨勪富瑕侀棶棰樻槸璋冨害鍜屼笉纭畾鎬ф潈閲嶃€?* v3 鐨勬棫閰嶇疆鍑犱箮鍏ㄩ潰瑙﹀彂寮傚父锛泇4 淇璋冨害鍚庯紝lambda=0/1 绋冲畾锛岃€?lambda=10 鏄庢樉鎹熷閲嶅缓銆?5. **鐭▼ exploration 閫傚悎绛涢敊鍜岀瓫缁撴瀯锛屼笉鏇夸唬姝ｅ紡璁粌銆?* 鏈€缁堟寮忛厤缃簲鑷冲皯瀵瑰€欓€夌偣鍋氬 seed銆佸畬鏁磋缁冮绠楀拰缁熶竴瑙ｇ爜璇勪及锛涘挨鍏朵笉搴斾緷鎹崟娆?50-epoch 鎺㈤拡涓皬浜庣害 0.5 dB 鐨勫樊寮傜洿鎺ュ畾妗堛€?
## 7. CSV 瀛楁璇存槑

- `version/group/family/stage/size_label/profile/target`锛氬疄楠岃韩浠藉拰鍒嗙粍銆?- `experiment_purpose`锛氳閰嶇疆鎵€灞炲疄楠岀粍鐨勭洰鐨勩€?- `initial_psnr_db/peak_psnr_db/final_psnr_db`锛氶涓€佹渶楂樺拰鏈€鍚庝竴涓浐瀹氭帰閽堢粨鏋溿€?- `gain_from_initial_db/drop_from_peak_db`锛氱煭绋嬫敹鐩婁笌璁粌鍥炶惤銆?- `needs_attention/attention_reason/validation_basis`锛氬紓甯告爣璁般€佸師鍥犲拰鍒ゅ畾鏉ユ簮銆?- `trajectory`锛?銆?0銆佲€︺€?0 epoch-equivalent 鐨勫畬鏁?PSNR 杞ㄨ抗銆?- `data_source/metrics_or_log_path`锛氭瘡椤圭粨鏋滅殑鏈湴璇佹嵁浣嶇疆銆?- `rmdsrn_final_*`锛歷4 RMDSRN 鏈€鍚庝竴姝ユ棩蹇椾腑鐨勮缁冩崯澶变笌璋冨害鐘舵€併€?- `result_summary`锛氭瘡椤归厤缃殑涓€鍙ヨ瘽涓枃缁撴灉銆?
## 8. 涓昏鏈湴鏉ユ簮

- `scripts/ablation/generate_architecture.py`銆乣scripts/ablation/summarize_architecture.py`
- `EXPLORATION_V2.md`銆乣scripts/sensitivity/generate_routing_and_depth.py`
- `EXPLORATION_V3.md`銆乣scripts/exploration/generate_rd_curve_smoke.py`銆乣scripts/exploration/summarize_rd_curve_smoke.py`
- `EXPLORATION_V4.md`銆乣scripts/ablation/generate_depth_and_regularization.py`銆乣scripts/ablation/summarize_depth_and_regularization.py`
- `batch_logs/exploration*/.../status.tsv`銆乣exploration_summary.tsv`銆乣needs_attention.txt` 涓?v4 鐨?81 浠芥棩蹇?