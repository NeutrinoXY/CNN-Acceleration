import functions as f
import layers as l

#f.Conv2D([1,2,3,4,5,6,7],3,1,0,1,0)
#f.Conv2D([1,2,3,4,5,6,7],3,2,0,1,0)

from PIL import Image
import numpy as np
#weights1=[[0.03,0.02,0.04,0.01],[0.02,0.03,0.03,0.02],[0.02,0.05,0.02,0.01],[0.01,0.01,0.03,0.05]]
weights2=[[0.1,0.2],[0.3,0.35]]
biases1=[-0.96178979, -0.23917986, -0.23212402, -0.2245342,  -1.71852744, -0.25244606,
 -0.70807356, -2.07269645, -0.40352401, -0.65268344, -1.05090392, -0.08804913,
 -0.51288158, -0.19405423, -0.62806863, -1.99370205, -0.1056308,  -1.53708804,
 -1.42091131, -2.4368856,  -0.81119192, -0.52126473, -2.35101843,  0.10267098,
 -0.17425123, -0.34821951, -1.04184127, -0.47194067, -0.25883618, -0.93448347,
 -0.3976385,  -0.91926754, -0.37985542, -1.39156711, -2.16277623, -0.39628947,
 -0.43708149, -0.66539854, -0.1934417,  -0.75807929, -1.36432159, -0.74310488,
 -0.57619452, -1.80293298, -0.38207579, -0.95824027, -0.23958017, -1.23103976,
 -0.84340107, -1.08316875, -0.19093898, -0.26658115, -0.69363737, -1.06321299,
 -0.18085641, -0.48473349, -1.3118434,  -1.0116632,  -0.45440874, -1.69262898,
 -2.28522992, -0.32850105, -0.29507414, -0.16844949]
weights1=[[[[  7.76161611e-01,  -2.48498544e-01,   1.60512999e-01,   8.50527585e-01,
    -1.34358436e-01,   2.22376049e-01,   1.81620896e-01,   1.52826503e-01,


     1.34166867e-01,   9.62008610e-02,   4.56986539e-02,  -1.08104669e-01,


     4.64021564e-01,   1.03624657e-01,   2.11331755e-01,  -4.68664974e-01,


     1.91000968e-01,   3.34794462e-01,   1.08155064e-01,   2.53502578e-01,


     3.68827283e-01,  -2.86423177e-01,   3.64328504e-01,  -1.82979569e-01,


    -2.24659115e-01,  -8.59970152e-02,   5.67903101e-01,  -8.84378314e-01,


     7.82794952e-02,  -6.15172535e-02,  -1.92909949e-02,  -2.99065948e-01,


     2.88093448e-01,  -1.81038007e-01,  -6.79351091e-01,  -6.99419677e-01,


    -2.35205591e-01,  -4.24594849e-01,  -1.15093899e+00,  -3.33046377e-01,


    -2.83903480e-02,   1.38027295e-01,  -4.25205976e-01,  -1.65925622e-01,


     5.33015616e-02,   5.24923921e-01,   1.47039250e-01,  -7.33614743e-01,


    -1.13636680e-01,   7.24750638e-01,   4.86624628e-01,   2.63595402e-01,


    -5.39344490e-01,   1.89281926e-01,   4.70552921e-01,   2.82166719e-01,


    -4.30119514e-01,  -4.59248364e-01,  -1.00340120e-01,  -8.12314510e-01,


     6.81686759e-01,   2.52034932e-01,   1.88880607e-01,   1.34451225e-01],


  [ -3.75436038e-01,   1.29757226e-01,  -6.71234488e-01,  -8.19130465e-02,


     3.53851765e-01,   3.49978298e-01,  -4.88686204e-01,   2.69330323e-01,


     1.32539779e-01,   6.51805401e-02,   1.20083869e-01,   8.54424164e-02,


     6.05659820e-02,   3.90102684e-01,   3.14871728e-01,  -1.05442131e+00,


     5.90656042e-01,   4.77782935e-01,   1.03091255e-01,   1.40705064e-01,


    -5.70846140e-01,  -1.76538676e-02,  -5.82785547e-01,   2.10263386e-01,


     4.57562417e-01,   6.45783067e-01,   2.43667830e-02,  -9.18642059e-02,


     1.53875411e-01,  -3.87997597e-01,  -2.32485950e-01,   1.64460033e-01,


    -1.30670995e-01,   3.59582722e-01,  -2.16383234e-01,   3.62844169e-01,


     2.67562777e-01,   6.96346879e-01,   9.46552336e-01,   5.91474891e-01,


    -2.30942056e-01,   3.19015644e-02,  -1.37264520e-01,  -1.81452230e-01,


    -5.38357079e-01,   3.18798721e-01,  -4.11576331e-01,   1.46301255e-01,


    -5.57499766e-01,   6.85754776e-01,  -1.56995565e-01,   1.37383699e-01,


     5.07644236e-01,  -2.87918180e-01,  -1.15080583e+00,   2.66000718e-01,


    -8.29896051e-03,   1.87151015e-01,  -9.31538790e-02,   1.27310336e-01,


     2.33127564e-01,  -2.63439596e-01,   3.20874117e-02,   2.38248363e-01],


  [ -5.90038419e-01,  -2.20828012e-01,   1.85186580e-01,  -7.65575230e-01,


     6.46746457e-01,  -6.80413187e-01,  -3.25711421e-03,  -1.30769327e-01,


    -1.07530113e-02,   5.58659136e-02,  -7.08921254e-01,   1.33266017e-01,


    -2.07169175e-01,   1.01410113e-01,   4.01673377e-01,  -1.15898693e+00,


    -2.47530147e-01,  -6.18088484e-01,   2.77952284e-01,  -4.11564648e-01,


    -4.96349752e-01,  -3.30113322e-01,   1.31404102e-01,   9.58177596e-02,


    -1.68185726e-01,  -8.22737575e-01,  -2.18517080e-01,   7.51179829e-02,


    -8.69018316e-01,  -1.81116894e-01,   1.33367315e-01,  -2.13091746e-01,


     6.15926012e-02,   1.07851341e-01,   7.93644905e-01,   6.12991571e-01,


     3.01920652e-01,  -4.16269749e-01,   1.37577012e-01,  -1.10169679e-01,


    -1.76870361e-01,  -3.77542704e-01,  -1.55226633e-01,  -5.77605426e-01,


    -3.84085029e-01,  -2.00456545e-01,   3.69627476e-01,   3.57097596e-01,


    -1.29326463e-01,   5.12910903e-01,   3.62121344e-01,  -1.05215229e-01,


     1.16793150e-02,  -6.66345954e-02,   7.71256208e-01,  -5.71562529e-01,


    -1.38562799e-01,  -2.64885813e-01,  -9.83139873e-02,   2.77395099e-01,


     2.54172564e-01,  -1.49430595e-02,   1.31984979e-01,  -3.15992177e-01]],







 [[  9.60672736e-01,   5.32902658e-01,   1.82326347e-01,  -7.35485926e-02,


    -2.27705017e-01,   2.01436713e-01,  -8.95649660e-03,   4.67083484e-01,


    -6.85884506e-02,  -6.39785826e-01,  -1.55492291e-01,   3.86742800e-02,


     9.14473295e-01,   4.10686314e-01,   1.70327917e-01,   7.59450257e-01,


    -6.61756039e-01,   3.74272764e-01,  -3.81361634e-01,   2.60740548e-01,


     2.94260502e-01,  -3.86789829e-01,   7.76772648e-02,  -3.05448174e-01,


    -1.24949954e-01,   7.50285625e-01,   6.76995516e-01,   1.09962547e+00,


    -2.43140578e-01,   7.38672912e-02,   6.72177613e-01,   5.47048271e-01,


    -9.28277075e-01,   2.75409043e-01,  -3.65742862e-01,   4.71677125e-01,


    -8.13470364e-01,  -2.49666333e-01,  -1.64769435e+00,  -1.26908195e+00,


    -9.56639722e-02,   6.28947020e-02,  -4.68298048e-01,   1.49155766e-01,


     2.01655835e-01,  -1.06976144e-01,   1.12327349e+00,  -5.27854383e-01,


    -7.51667693e-02,  -8.88901412e-01,  -1.95564672e-01,  -2.76372373e-01,


    -4.97104824e-01,   8.85807395e-01,   3.60117257e-01,  -2.05392405e-01,


     1.41184973e-02,  -3.91044199e-01,  -8.11052024e-02,  -8.29435170e-01,


    -1.34061500e-01,   8.72895718e-01,   2.39753649e-01,   3.33602965e-01],


  [ -2.35546052e-01,   2.77748853e-01,  -9.75284800e-02,  -3.61909047e-02,


    -4.04979438e-02,   3.62943858e-01,  -1.09191529e-01,   4.26069826e-01,


     7.76030272e-02,   5.33522427e-01,   9.67009738e-02,  -5.36525905e-01,


     4.69916552e-01,  -3.02897543e-01,   1.60668850e-01,   2.24909082e-01,


    -5.00959575e-01,   2.97648817e-01,  -1.97794065e-01,   1.63773388e-01,


    -4.94411796e-01,  -1.01911688e+00,  -7.02163577e-01,  -5.20233288e-02,


     2.76713103e-01,   2.66480029e-01,  -8.59691296e-03,   1.99906215e-01,


     6.81091964e-01,   2.64081284e-02,   3.71331096e-01,   1.09643722e+00,


    -4.47812267e-02,   2.23103061e-01,  -4.29001719e-01,  -1.72720075e-01,


    -3.20626944e-01,   7.75874794e-01,   1.14361751e+00,   4.69817251e-01,


    -3.76006275e-01,  -3.39897454e-01,  -1.28815815e-01,  -2.49098599e-01,


     1.54276162e-01,  -1.34659708e-01,  -1.05381191e+00,   3.27905953e-01,


    -1.62469111e-02,  -1.15243769e+00,  -1.43943653e-02,  -7.58939803e-01,


     4.93636936e-01,  -6.58993661e-01,  -3.82907093e-01,  -1.21142909e-01,


     2.65080392e-01,   1.50004938e-01,  -3.20198715e-01,   1.78498000e-01,


    -4.10050124e-01,  -1.48561463e-01,   1.87041059e-01,  -9.63920131e-02],


  [ -1.02304041e+00,   2.14483440e-01,   1.81453317e-01,   1.45611733e-01,


    -3.38225007e-01,  -5.69852173e-01,   1.79515891e-02,   2.16746598e-01,


    -6.56729102e-01,   4.22111183e-01,  -4.45417076e-01,  -4.30303905e-03,


    -5.12185514e-01,   3.55677873e-01,   4.11949188e-01,   2.78784391e-02,


    -7.40455925e-01,  -5.53836048e-01,   3.19318503e-01,  -3.40898514e-01,


    -3.20197403e-01,   3.47443849e-01,  -2.60100096e-01,   1.52353883e-01,


     3.72100681e-01,  -5.60707271e-01,  -5.20809218e-02,   3.22161168e-01,


    -3.67547393e-01,   4.39447910e-01,   1.14972457e-01,   1.51062489e-01,


    -3.98898989e-01,   5.77800095e-01,   4.11622018e-01,  -3.75474930e-01,


     1.21243801e-02,  -2.04587609e-01,   5.29292285e-01,   6.61590517e-01,


    -2.38518462e-01,   3.91001433e-01,  -5.95473424e-02,  -3.44650626e-01,


     6.48552850e-02,   2.68810302e-01,  -8.31648469e-01,   1.12196118e-01,


     5.64706624e-01,  -4.61417317e-01,  -6.17975771e-01,   1.88048214e-01,


    -3.14943977e-02,  -1.58291385e-01,   3.46839353e-02,   4.66801643e-01,


     2.62041658e-01,   7.41179613e-03,  -4.26963240e-01,   3.16649675e-01,


    -3.03269550e-02,   6.27339661e-01,  -6.51236922e-02,  -5.73934793e-01]],







 [[  5.01330853e-01,  -1.14559136e-01,  -3.64792466e-01,  -3.60680759e-01,


    -2.09404349e-01,   1.56536505e-01,  -1.29744604e-01,   1.35364801e-01,


    -4.56306458e-01,  -2.71533459e-01,   2.19290286e-01,   2.07431376e-01,


     7.02628016e-01,  -4.38615084e-01,  -1.04644291e-01,  -3.07850968e-02,


     4.36426699e-01,  -5.72431743e-01,  -1.03087336e-01,   4.54341531e-01,


     1.17766045e-01,   1.13421226e+00,  -3.29544172e-02,  -2.55522370e-01,


     3.79672438e-01,   7.02002347e-02,  -1.35271937e-01,  -5.50173521e-01,


    -1.47581100e-01,   5.78781627e-02,  -6.68457866e-01,  -5.59718966e-01,


    -1.64417997e-01,   9.08076577e-03,   2.50503689e-01,   1.00307655e+00,


     3.51903528e-01,  -2.79172927e-01,  -9.59753156e-01,   4.03979599e-01,


     2.22488455e-02,  -5.19331135e-02,  -3.76353085e-01,   7.40053535e-01,


     1.51055649e-01,  -5.44252157e-01,  -4.18166548e-01,   9.73225161e-02,


    -1.28041640e-01,   1.04387295e+00,  -1.26839459e-01,   3.25017005e-01,


    -6.07173502e-01,   2.04142421e-01,   5.50497055e-01,  -8.36751282e-01,


    -3.44051182e-01,   8.68117452e-01,  -7.49045908e-02,  -7.33813882e-01,


    -8.57287288e-01,  -1.23791896e-01,   5.31779289e-01,   2.85947531e-01],


  [ -1.45647064e-01,  -1.38431922e-01,   4.18289959e-01,   1.18129097e-01,


     1.86355874e-01,   6.55830801e-01,  -2.07997873e-01,   5.82006931e-01,


     1.40560657e-01,   6.66349053e-01,  -2.14739263e-01,  -5.11870384e-01,


     4.13589925e-02,  -2.98726529e-01,  -1.77779660e-01,   4.58390355e-01,


     3.62615824e-01,   2.75018126e-01,   1.15465343e-01,   1.33331999e-01,


     9.03377077e-04,   2.97779649e-01,  -5.21559417e-01,   6.03706501e-02,


    -6.51767075e-01,   3.61036837e-01,  -6.36295080e-02,  -3.32743347e-01,


     2.77055502e-01,   4.29438084e-01,   5.67676350e-02,  -6.88454285e-02,


     1.21350622e+00,  -1.04266989e+00,  -2.30408356e-01,  -2.66690731e-01,


     8.09086263e-02,   2.88938761e-01,   8.79877865e-01,  -1.11357175e-01,


    -2.95652181e-01,  -3.25807571e-01,  -4.82368805e-02,  -2.69714832e-01,


     4.82096910e-01,  -5.26125968e-01,   4.66057092e-01,   1.32477984e-01,


    -4.66865629e-01,  -1.17039733e-01,   1.74472630e-01,   1.11710645e-01,


     1.20943761e+00,  -6.16601348e-01,  -1.46502197e+00,  -1.84567779e-01,


    -2.19721243e-01,   3.04644734e-01,  -1.73144415e-01,  -6.43757805e-02,


    -7.72365212e-01,  -2.42709979e-01,   5.65523446e-01,  -8.23019072e-05],


  [ -9.90824044e-01,  -1.48276687e-01,  -3.62250239e-01,   2.94857174e-01,


     2.98288018e-01,  -7.35022664e-01,   3.23955238e-01,  -1.84891701e-01,


    -2.54315108e-01,  -2.64873207e-01,   5.19183338e-01,   9.57279205e-02,


    -7.11992204e-01,  -5.27525961e-01,   3.86652678e-01,  -7.58771151e-02,


     5.44695079e-01,   2.06998050e-01,   2.57141769e-01,  -4.70112234e-01,


     3.61549884e-01,   2.79776961e-01,  -4.75023806e-01,   4.94600683e-01,


    -3.17675591e-01,  -2.89018810e-01,  -1.23553127e-01,   1.16274588e-01,


     1.86852649e-01,   2.45654002e-01,  -3.09276879e-01,  -1.44187614e-01,


    -1.57076210e-01,  -4.35139090e-01,   2.77765870e-01,  -1.10522127e+00,


    -2.35151485e-01,  -2.64939129e-01,   1.35814101e-01,  -3.42437655e-01,


    -6.00076206e-02,   3.30372930e-01,  -9.67702419e-02,  -1.96080923e-01,


    -1.89625874e-01,   3.27185750e-01,   1.87714353e-01,   8.76308158e-02,


     1.16612092e-01,   8.04388165e-01,   7.90634155e-02,  -3.77079993e-02,


    -5.04665315e-01,   3.59725624e-01,   8.75133276e-01,   6.31069124e-01,


    -1.10618524e-01,  -1.21162541e-01,  -3.11918259e-01,   4.34763640e-01,


    -5.39720953e-01,   3.45025688e-01,  -1.60519481e-01,  -7.37369597e-01]]],












[[[ -8.39314014e-02,   8.50490451e-01,   3.85195374e-01,   1.10835779e+00,


    -2.92097896e-01,  -3.74502093e-01,  -1.03811413e-01,   1.72141477e-01,


     1.46033943e-01,   5.38293600e-01,   2.06740186e-01,  -3.42795819e-01,


    -9.82701838e-01,   3.43194574e-01,  -6.27951801e-01,   1.11788310e-01,


    -7.34225154e-01,   1.64072290e-01,  -2.97290921e-01,  -2.79023871e-03,


     3.85548323e-01,   4.11924154e-01,  -7.95998350e-02,  -3.12559456e-02,


    -3.00638020e-01,   1.02340710e+00,   5.72679222e-01,  -5.95233083e-01,


    -5.23255020e-02,   4.41446118e-02,  -7.79995799e-01,  -4.54478621e-01,


     7.80953765e-01,  -2.91712970e-01,   2.02445865e-01,  -8.95194590e-01,


    -9.35130119e-01,  -2.54428506e-01,   5.81861973e-01,  -8.20958734e-01,


    -2.23566696e-01,   2.65160240e-02,   3.48954976e-01,  -2.08795398e-01,


    -1.51536584e-01,   2.80171096e-01,   3.46150607e-01,  -6.10858321e-01,


    -1.04617616e-02,  -3.26152146e-01,   1.35902181e-01,   1.54084321e-02,


     9.62105691e-01,  -3.68852407e-01,   3.47195379e-03,   6.35857761e-01,


    -5.02433300e-01,  -4.54378754e-01,   4.07426581e-02,  -3.62382203e-01,


    -7.85024315e-02,   3.70484889e-02,  -2.47100547e-01,  -6.85565770e-01],


  [ -1.98174760e-01,   6.56847954e-01,  -4.40995336e-01,  -4.08877790e-01,


    -2.33131021e-01,  -5.48370063e-01,  -1.47920385e-01,   1.31161943e-01,


     4.76626679e-03,  -7.62621284e-01,   1.84426010e-01,   1.69315166e-03,


    -3.26390535e-01,  -3.30912739e-01,  -4.65652257e-01,  -1.58172667e-01,


    -3.69473398e-01,   1.79663718e-01,  -1.06662497e-01,   3.91394049e-02,


     8.28735888e-01,   1.05208087e+00,  -4.09567684e-01,   2.58130342e-01,


     4.63790983e-01,   5.69822609e-01,  -1.07324429e-01,   2.94342071e-01,


     3.93737346e-01,   1.77944973e-01,  -5.85602880e-01,  -2.05351993e-01,


    -3.36021721e-01,  -1.31838899e-02,   1.05070472e-01,   2.68303841e-01,


    -4.78039503e-01,   2.66509891e-01,  -1.06995851e-01,   2.67934829e-01,


    -1.38775287e-02,  -2.40255117e-01,   1.25734031e-01,   6.01991773e-01,


    -2.31250435e-01,   2.72105217e-01,  -3.09412092e-01,   8.07564378e-01,


    -1.29745647e-01,  -3.84858698e-02,  -2.36625560e-02,   8.62758398e-01,


    -1.46686268e+00,   4.93435562e-02,  -3.26102614e-01,   2.16670677e-01,


     1.57359131e-02,   6.81126416e-02,   3.72261286e-01,   2.90373266e-01,


    -1.28661335e-01,   2.02999607e-01,  -4.91280079e-01,   3.49493325e-01],


  [  3.43951970e-01,   4.21520531e-01,   9.16078806e-01,  -8.19815636e-01,


     3.98557514e-01,   1.01633596e+00,   8.56090635e-02,   9.66312177e-03,


     3.67716372e-01,  -5.82232893e-01,  -7.87472725e-01,   8.17951411e-02,


     7.39598989e-01,   3.65377128e-01,   5.60632586e-01,   3.98833066e-01,


    -7.10866570e-01,  -6.76821947e-01,   2.84636527e-01,  -5.39649371e-03,


     5.07831454e-01,  -8.45393538e-02,  -1.54830024e-01,  -3.44370026e-03,


    -7.12392777e-02,  -9.43199158e-01,   2.13039778e-02,  -4.14042860e-01,


    -5.65283656e-01,   4.23906714e-01,  -4.48011458e-01,  -2.17172161e-01,


     4.16211337e-01,  -3.39933485e-01,  -5.08899540e-02,   1.01104128e+00,


    -1.30691737e-01,  -1.44198433e-01,  -4.16877806e-01,   5.01062512e-01,


    -2.91021466e-01,   2.40867779e-01,   3.61072481e-01,   3.73241723e-01,


    -4.33666743e-02,   3.75701219e-01,  -2.61058003e-01,   6.87985539e-01,


    -2.78707981e-01,  -6.56589270e-01,   7.67194748e-01,   1.52717367e-01,


     3.59449893e-01,  -1.16981074e-01,   2.38495275e-01,  -6.49018228e-01,


    -2.14116052e-02,   8.41672793e-02,   4.06660616e-01,   2.03296810e-01,


    -6.42205179e-02,  -1.79008096e-01,  -2.05780342e-01,   4.71821934e-01]],







 [[  8.49607587e-02,  -9.07708168e-01,  -4.68360633e-02,  -6.78681195e-01,


     2.55590022e-01,  -3.70737493e-01,  -2.47648975e-04,  -1.26542735e+00,


     6.81099355e-01,  -1.03413679e-01,   6.19207062e-02,   1.28259733e-01,


    -6.61480963e-01,  -7.79025495e-01,  -4.41616356e-01,   1.97218135e-01,


     1.14160645e+00,   2.65112638e-01,  -2.45277449e-01,  -6.45792484e-02,


     2.39898309e-01,  -6.39815748e-01,  -1.68126926e-01,  -7.97960311e-02,


    -1.50880620e-01,   4.75938506e-02,  -6.03342727e-02,   1.04877460e+00,


    -2.97589362e-01,   1.87879741e-01,   3.37664515e-01,  -1.23740114e-01,


     5.50057590e-01,   5.24069965e-02,   4.12026167e-01,  -2.67650217e-01,


     4.54048455e-01,  -5.47881722e-01,   3.20952535e-01,  -7.53788292e-01,


     1.97546855e-01,   2.35512387e-02,   6.58017024e-02,  -5.38976848e-01,


     2.46955976e-02,  -3.65637511e-01,   4.40421999e-01,   8.42338130e-02,


     2.70614713e-01,   3.63510758e-01,  -6.77410290e-02,  -6.04095817e-01,


     5.21602809e-01,   1.10905540e+00,  -8.44358742e-01,   1.58839837e-01,


    -7.30358899e-01,  -4.72836316e-01,   1.04609162e-01,  -4.58588868e-01,


     2.28624389e-01,   7.10067078e-02,  -5.01374424e-01,  -3.07364017e-01],


  [ -7.70766437e-02,  -8.51032734e-01,  -3.38616192e-01,   1.11070881e-02,


     3.06193888e-01,  -3.59684050e-01,   1.03236184e-01,  -1.36560237e+00,


     1.24922022e-01,  -1.60889793e-02,  -4.31125462e-02,   2.21886169e-02,


    -1.89857572e-01,  -8.54053020e-01,  -5.09496033e-01,   3.39507818e-01,


     9.03808296e-01,  -2.29436100e-01,  -5.56389242e-02,   4.74885292e-02,


     3.95104438e-01,  -4.37285960e-01,   2.15648621e-01,   8.17479715e-02,


     3.32108259e-01,  -4.04974073e-01,  -7.01267943e-02,   4.31280643e-01,


     7.82101989e-01,   3.30681652e-01,  -9.77516174e-02,   1.90953672e-01,


     9.14848521e-02,   5.78370333e-01,   2.13261068e-01,  -1.16384581e-01,


    -1.26889735e-01,   1.20002292e-01,   1.23242545e-03,  -4.42939773e-02,


     3.34843248e-01,  -6.20182872e-01,   1.28590077e-01,  -5.93849309e-02,


     2.04506487e-01,  -3.41673434e-01,  -4.13799673e-01,   4.23467070e-01,


     5.31581044e-01,   5.72873175e-01,  -1.35969624e-01,  -7.78557062e-01,


    -1.06611109e+00,  -3.58787805e-01,   2.00035667e+00,  -2.60448307e-01,


    -1.05338089e-01,  -3.51264216e-02,   7.34841764e-01,   2.93688029e-01,


     2.18286831e-02,  -1.62231147e-01,  -3.58803958e-01,   1.27179429e-01],


  [ -2.11967528e-01,  -3.12666774e-01,  -6.16052806e-01,   7.22914159e-01,


     3.46881628e-01,   9.76357043e-01,   2.05345452e-01,  -5.53850830e-01,


     1.26264140e-01,   3.10062051e-01,  -3.03854674e-01,   2.69494236e-01,


     4.70229805e-01,  -5.90186417e-01,  -4.73576896e-02,   6.00809395e-01,


     1.48498130e+00,  -6.61907196e-01,   4.49461699e-01,  -1.36012346e-01,


    -1.86260059e-01,   8.93208012e-02,   1.02060121e-02,  -8.60862955e-02,


     5.59054554e-01,   2.97558844e-01,   1.91789523e-01,   6.92141429e-02,


     5.01939833e-01,   3.34639162e-01,   5.63041925e-01,  -1.15355156e-01,


     4.47624743e-01,   5.54339170e-01,  -2.70483017e-01,   3.51312071e-01,


     1.07745364e-01,  -1.21441193e-01,  -3.01759452e-01,   7.44077027e-01,


     3.35990399e-01,   8.49242687e-01,   4.11609113e-01,  -5.55416718e-02,


     2.54916668e-01,   1.07258379e-01,  -4.17649537e-01,  -2.42646853e-03,


     5.19543350e-01,   3.66005868e-01,  -5.41666687e-01,  -1.97139665e-01,


     3.85661215e-01,  -3.97062600e-01,  -1.27104723e+00,   1.02461956e-01,


    -2.96710759e-01,   1.80854484e-01,   4.97205049e-01,   1.54857963e-01,


     4.85793769e-01,   7.80765191e-02,   1.57921091e-01,   4.78809536e-01]],







 [[ -2.36998945e-01,  -4.84563261e-02,  -3.92562687e-01,  -7.49755681e-01,


    -2.42232591e-01,  -2.12868825e-01,  -3.47914617e-03,   1.34789780e-01,


     1.36496693e-01,  -1.00403714e+00,   1.80319503e-01,   2.04997003e-01,


     6.67608440e-01,   3.51606578e-01,  -1.21874526e-01,  -6.24105871e-01,


    -5.51271737e-01,  -9.84289274e-02,   1.25596195e-01,   4.31026854e-02,


    -3.19206178e-01,  -2.67196327e-01,  -4.93190959e-02,  -9.14065167e-02,


     4.77470368e-01,  -2.55972445e-01,  -1.02909005e+00,  -4.67924386e-01,


    -1.78460732e-01,  -7.75025487e-02,   5.61712205e-01,  -2.29411811e-01,


    -1.11717439e+00,  -2.32459605e-01,   3.59467447e-01,   8.80717158e-01,


     7.70880401e-01,   1.17717050e-01,   2.53472626e-01,   1.16398036e+00,


    -2.54576862e-01,   6.12003691e-02,   4.72067520e-02,   2.89457321e-01,


    -1.19788498e-01,  -6.71028122e-02,  -1.03446269e+00,   4.22202677e-01,


     8.90363082e-02,   2.69086450e-01,  -1.12975985e-02,   2.12425634e-01,


    -3.66557568e-01,   5.19811869e-01,  -2.82413870e-01,  -6.87421322e-01,


    -2.30669796e-01,   6.81256294e-01,   1.24732025e-01,  -4.14800823e-01,


     5.00665605e-01,   5.26225790e-02,  -6.92254126e-01,  -1.32519871e-01],


  [  1.72998577e-01,   3.31506543e-02,   1.21986139e+00,   4.26982552e-01,


     1.61740556e-01,   4.47120611e-03,  -1.48252070e-01,   4.90492135e-02,


    -1.77327201e-01,   9.35278714e-01,  -3.36960256e-01,  -2.57189453e-01,


     1.68453440e-01,   1.06070960e+00,  -3.41527462e-01,   9.80788618e-02,


    -8.27348292e-01,   1.53600886e-01,  -1.27273455e-01,   9.80154648e-02,


     2.05945186e-02,  -9.74990368e-01,   5.45290768e-01,  -5.74391820e-02,


    -6.51707649e-01,  -2.38011092e-01,  -1.32892996e-01,  -1.10654451e-01,


    -4.55272734e-01,  -1.08412795e-01,   4.56548125e-01,  -1.65275857e-01,


     9.90664735e-02,  -2.93421298e-01,  -1.92705151e-02,  -1.90186888e-01,


     9.97857824e-02,   6.44135401e-02,  -1.63728058e-01,  -5.82443893e-01,


    -9.46531072e-02,  -4.46330905e-01,   1.04590222e-01,  -2.82025427e-01,


     2.14839637e-01,  -1.00420870e-01,   1.12457371e+00,  -3.42547476e-01,


    -2.09245220e-01,  -6.05833352e-01,   1.91030845e-01,   6.44876063e-01,


     6.31038845e-01,  -5.04218876e-01,   3.78403872e-01,  -1.74259275e-01,


     3.38188782e-02,  -1.72861218e-01,   7.24068522e-01,   9.01121870e-02,


    -6.49800822e-02,  -6.70856014e-02,  -7.24537075e-01,  -9.99592431e-03],


  [  1.66397199e-01,  -9.10729915e-02,  -3.72283220e-01,   3.62815440e-01,


     5.06182671e-01,   1.28087223e-01,  -7.58072187e-04,  -5.35163749e-03,


    -8.78584087e-01,   8.81229997e-01,   2.66331106e-01,   4.73420918e-01,


    -3.21976304e-01,   1.99918717e-01,  -2.89283749e-02,  -7.26074725e-02,


    -3.81174594e-01,   1.32324368e-01,  -3.63823175e-01,  -9.13418755e-02,


     2.11920455e-01,   3.26001495e-01,   4.67592254e-02,   6.71600504e-03,


    -3.25132966e-01,   3.25805426e-01,   5.93109243e-02,  -3.03219318e-01,


     4.78769839e-01,  -4.70736593e-01,  -9.97348800e-02,  -1.28129795e-01,


    -7.08665729e-01,   3.70769389e-02,   4.06381860e-02,  -9.20610189e-01,


     8.24693367e-02,   2.50798047e-01,  -1.05889007e-01,  -3.03918183e-01,


     1.40190154e-01,   3.73602211e-01,   2.97531366e-01,  -5.38713075e-02,


     6.47297427e-02,  -2.44661018e-01,   6.73320591e-01,  -5.56879163e-01,


    -1.09566964e-01,  -2.92503655e-01,  -1.91087082e-01,   1.47529960e-01,


    -3.33553284e-01,  -2.19735518e-01,  -1.15406133e-01,   7.33080685e-01,


     6.90555125e-02,  -3.26769978e-01,   5.52241206e-01,   2.00050324e-01,


     1.08331695e-01,   6.29022121e-02,  -1.92151010e-01,   1.56939760e-01]]],












[[[ -7.01947451e-01,  -2.22118035e-01,  -6.64498329e-01,   1.37038946e+00,


     1.92634299e-01,  -5.28255403e-02,  -5.66490553e-02,   3.54932934e-01,


    -4.39775079e-01,   7.61640906e-01,  -1.82393327e-01,  -3.00388671e-02,


    -1.84044555e-01,  -7.74561316e-02,   3.07325363e-01,  -1.19302245e-02,


     2.75343746e-01,  -2.19595850e-01,  -4.03575987e-01,  -3.46808970e-01,


    -1.06062388e+00,  -2.24258274e-01,  -1.97178242e-03,  -3.39935154e-01,


    -3.21310163e-01,  -5.20497680e-01,   3.18997741e-01,  -2.43050188e-01,


     2.78782248e-01,  -1.66213676e-01,   3.70340943e-01,   8.58094320e-02,


    -4.64572728e-01,  -1.04681499e-01,   8.83560896e-01,  -2.24431470e-01,


     6.43306971e-01,   3.23283404e-01,   7.65596330e-01,  -1.89472198e-01,


    -4.98158634e-01,   2.95967627e-02,   5.51152170e-01,  -8.67039636e-02,


    -1.15243331e-01,  -1.05818033e+00,   3.14451777e-03,  -6.85959399e-01,


    -2.21075118e-01,  -3.83035868e-01,  -8.81358013e-02,   1.79749385e-01,


     1.19058561e+00,  -1.08676863e+00,   7.50292987e-02,   6.04668796e-01,


     5.37994862e-01,  -5.25170803e-01,   9.09130946e-02,   3.76956493e-01,


    -2.27690518e-01,  -2.68424660e-01,   2.53046215e-01,  -5.61369956e-01],


  [  1.13085106e-01,  -2.71598250e-01,  -1.28876436e+00,  -6.75436616e-01,


    -3.31896186e-01,  -6.71005666e-01,  -1.08678415e-01,   1.02761798e-01,


    -1.22705854e-01,  -4.50376958e-01,   1.35143712e-01,   5.74207418e-02,


     1.13173395e-01,  -7.07620144e-01,  -3.12802702e-01,   6.09535500e-02,


     4.72673953e-01,   2.59470522e-01,   4.56461273e-02,   2.41918210e-02,


    -5.60013764e-03,   2.70296574e-01,   2.53300555e-02,  -2.58142240e-02,


     5.11134028e-01,  -1.76833533e-02,  -4.51136418e-02,  -5.94680123e-02,


    -3.60474512e-02,   3.52540553e-01,   1.22776926e+00,   3.47622633e-02,


    -6.72763288e-01,  -1.13109238e-01,   5.70334315e-01,   1.76943079e-01,


    -1.42183691e-01,   3.12751621e-01,  -9.40591037e-01,  -3.65662500e-02,


    -3.24517727e-01,  -2.61223704e-01,   6.74458817e-02,   7.64670908e-01,


    -6.15565836e-01,  -4.29569036e-01,  -1.92659691e-01,   4.72649544e-01,


    -1.64455682e-01,  -9.76060927e-02,   4.51797210e-02,   6.02371633e-01,


    -1.72665870e+00,   8.84901166e-01,  -2.83802092e-01,   4.97397155e-01,


     2.72728473e-01,   1.70170039e-01,  -3.58055592e-01,  -1.74602624e-02,


     2.17387587e-01,   4.22749579e-01,   3.88979018e-01,   1.83312058e-01],


  [  6.89246237e-01,  -3.47034991e-01,   6.38105512e-01,  -8.74994516e-01,


    -7.35856473e-01,   6.74728632e-01,  -2.02624112e-01,   3.36832404e-02,


     2.65743375e-01,  -3.67538780e-01,  -3.38495165e-01,   4.24839795e-01,


     2.08996788e-01,  -1.72193184e-01,  -2.24915415e-01,   4.74378943e-01,


     5.33342004e-01,  -8.43673870e-02,   4.79539901e-01,   3.84356260e-01,


    -2.51581460e-01,  -5.42091668e-01,  -2.24852026e-01,  -1.21055484e-01,


    -2.14399159e-01,   3.25604707e-01,   1.33234430e-02,  -1.61677763e-01,


     5.57301752e-02,   7.87807629e-02,   1.89665779e-02,   2.73597836e-01,


    -2.96072476e-02,  -5.56614816e-01,  -7.66775489e-01,   3.49472106e-01,


     3.70076559e-02,   1.22809507e-01,   1.30966142e-01,   2.64788032e-01,


    -6.00397706e-01,   1.36414960e-01,  -8.84070024e-02,   2.20807731e-01,


    -1.76071778e-01,   1.08285852e-01,  -2.01059096e-02,   3.27090144e-01,


    -3.86478275e-01,  -3.12721550e-01,   4.32479888e-01,   9.59931687e-02,


     6.79270685e-01,   2.87932664e-01,   2.35490650e-01,  -7.31170475e-01,


     3.85998972e-02,   8.79336372e-02,  -1.33252695e-01,  -5.03735960e-01,


     5.82374707e-02,  -2.04715386e-01,  -7.24048615e-02,   1.30053267e-01]],







 [[ -4.98236030e-01,  -9.83766839e-02,   1.35372913e+00,  -9.28194165e-01,


     3.12444150e-01,  -3.47835094e-01,  -1.07802495e-01,  -4.59278747e-02,


    -1.92835748e-01,   7.58381665e-01,  -2.89142668e-01,  -1.63536984e-02,


    -8.92702758e-01,   2.87986606e-01,   4.29437220e-01,  -1.14383869e-01,


    -4.42438632e-01,  -2.69912720e-01,   5.45205891e-01,   2.43834794e-01,


    -2.93321669e-01,   4.39905643e-01,  -2.76968088e-02,  -1.96628086e-03,


    -3.50850314e-01,  -9.52708066e-01,  -5.79595864e-01,   6.73751652e-01,


    -7.29663298e-02,   8.61263946e-02,  -8.00394833e-01,  -2.19169393e-01,


     6.12899423e-01,  -5.35122678e-02,   4.23287541e-01,  -3.54453892e-01,


     8.84494841e-01,  -4.67785925e-01,   9.78213191e-01,   5.38790524e-01,


     1.87312469e-01,   1.15906343e-01,  -4.10694629e-02,  -2.52788961e-01,


     6.45690262e-02,  -1.77044168e-01,   3.62350941e-02,   1.19827785e-01,


     8.07894394e-02,   1.11356050e-01,  -1.14469729e-01,  -5.71951687e-01,


     7.52762735e-01,  -2.00711295e-01,  -9.52164054e-01,   4.04627264e-01,


     6.26779675e-01,  -6.85685992e-01,  -6.67517409e-02,   6.08513594e-01,


    -1.39287615e-03,  -1.23517863e-01,   1.06123742e-04,  -2.71069378e-01],


  [  3.68813097e-01,   1.92192756e-02,   3.32743227e-01,   1.56311557e-01,


    -2.62367189e-01,  -6.47856712e-01,  -1.85439050e-01,  -5.35936415e-01,


    -1.34233251e-01,  -7.25640595e-01,  -1.16198041e-01,  -2.72837728e-01,


    -1.66555211e-01,   6.58000708e-01,   4.32535201e-01,   1.54008687e-01,


    -9.48598027e-01,  -9.89383459e-02,   3.61473888e-01,   8.79385471e-01,


     3.39148581e-01,   1.06050181e+00,   6.32046819e-01,   2.95550406e-01,


     2.42791325e-02,  -9.39785182e-01,  -2.14816913e-01,  -6.95751160e-02,


    -3.82337302e-01,  -1.61926895e-01,  -1.06120443e+00,  -1.89300239e-01,


    -5.55233717e-01,   3.77833605e-01,   2.17921287e-01,  -4.03822586e-02,


     1.65920824e-01,  -2.76607811e-01,  -9.22502697e-01,  -3.52852076e-01,


     4.79696453e-01,  -5.29079378e-01,  -2.12921605e-01,   4.66656208e-01,


    -1.66924745e-01,   7.01081455e-02,   9.76279378e-02,  -3.21820825e-01,


     4.95936215e-01,   4.10553664e-01,   1.41181834e-02,  -1.00602472e+00,


    -1.29636443e+00,   1.76816523e-01,   2.31864357e+00,  -7.84565359e-02,


     3.14435542e-01,  -5.22728413e-02,  -6.22599602e-01,   4.12146701e-03,


     1.77362978e-01,   2.47361466e-01,   3.34531784e-01,   1.30648509e-01],


  [  5.80548346e-01,   3.63592774e-01,   4.44654346e-01,   1.03601825e+00,


    -9.27651286e-01,   8.78242016e-01,  -4.42397952e-01,   1.67038724e-01,


     5.44826806e-01,  -8.28801572e-01,   6.72246277e-01,   6.75371066e-02,


     8.13679516e-01,   3.70145172e-01,  -4.84265685e-01,   1.09878235e-01,


    -3.18102658e-01,   1.22725733e-01,   3.99752349e-01,   9.34584796e-01,


    -1.64605230e-01,  -3.59266102e-02,  -2.31428832e-01,   1.86564967e-01,


     7.81028390e-01,   1.25071049e+00,  -8.10165145e-03,   1.82007313e-01,


     5.71951926e-01,  -5.47272086e-01,  -3.97172391e-01,   1.76859632e-01,


     3.03003311e-01,  -1.66562796e-01,  -7.57523715e-01,   4.13823158e-01,


    -8.40165988e-02,  -1.53030366e-01,  -4.02853042e-02,  -1.39606133e-01,


     5.64785421e-01,   5.91510236e-01,  -4.58737940e-01,   2.67627776e-01,


     3.73271614e-01,  -2.32415237e-02,   3.03101838e-01,  -7.00811327e-01,


     5.40676296e-01,   4.68330026e-01,   7.90660754e-02,  -3.35994720e-01,


     5.73572934e-01,  -8.63559097e-02,  -1.49682832e+00,  -3.70595694e-01,


     5.23340777e-02,   3.55483592e-01,  -2.21618578e-01,  -5.08427799e-01,


     5.70721701e-02,  -6.89455569e-02,   1.81414127e-01,   6.28211498e-02]],







 [[ -8.31269681e-01,   2.53183335e-01,  -6.51181400e-01,  -5.22105336e-01,


    -9.23550576e-02,   4.82853472e-01,   1.10759310e-01,   2.50292033e-01,


     1.88678965e-01,  -2.83184022e-01,   1.43565327e-01,   1.28661111e-01,


     4.63108718e-02,  -2.25079551e-01,   1.65179983e-01,  -2.81657338e-01,


     3.46983820e-01,  -3.64058495e-01,   2.60077149e-01,  -3.60868305e-01,


    -3.63013923e-01,  -8.17280635e-03,  -1.24436744e-01,  -3.86485189e-01,


     5.74708045e-01,  -3.90405394e-02,  -2.96109527e-01,  -1.20875381e-01,


     4.57754493e-01,  -2.30434518e-02,   3.91837060e-01,   1.81298792e-01,


     3.38135988e-01,   4.81675118e-02,  -3.06270927e-01,   2.07710043e-01,


    -7.78902709e-01,   4.54762250e-01,   9.53372061e-01,   1.59976971e+00,


    -3.66701543e-01,   2.78140213e-02,   4.56389129e-01,  -2.83338010e-01,


    -9.13435593e-02,   9.69477713e-01,  -9.38594759e-01,   6.47927821e-01,


    -4.31700423e-02,  -2.73662537e-01,  -4.50648218e-02,   5.75075150e-01,


    -6.86544299e-01,  -4.61769611e-01,  -3.21847975e-01,  -3.75020921e-01,


     6.09598637e-01,   6.75728679e-01,   1.85467638e-02,   6.09634697e-01,


     3.77541065e-01,  -1.64849773e-01,   1.62537113e-01,  -2.26831779e-01],


  [  1.90903366e-01,   2.33487174e-01,   7.69053221e-01,   6.38241231e-01,


    -1.58755183e-01,  -2.67797977e-01,   4.17783819e-02,  -1.53195579e-02,


     8.32230374e-02,   1.62994385e-01,  -1.17434911e-01,  -4.74200934e-01,


    -2.58455992e-01,   2.01440305e-01,   4.18058217e-01,   1.25891984e-01,


     3.02005321e-01,   1.47291645e-01,  -5.04998326e-01,   5.40702529e-02,


    -1.20757986e-02,  -4.68531132e-01,   6.97421312e-01,   1.03887916e-01,


    -6.75309658e-01,  -2.69423485e-01,   3.18425655e-01,  -3.60110134e-01,


    -1.12669861e+00,  -3.78738105e-01,  -3.81881326e-01,  -2.04465598e-01,


     1.37656778e-01,   5.59607089e-01,  -2.08657250e-01,  -2.77364329e-02,


     4.37591262e-02,  -2.60777742e-01,  -8.91684532e-01,  -5.83630145e-01,


     1.17197894e-01,  -3.01857024e-01,   2.32048005e-01,  -4.06395197e-01,


     3.30182791e-01,   7.75682688e-01,   8.00243139e-01,  -3.22807938e-01,


    -3.16507876e-01,  -2.55825907e-01,  -1.39141575e-01,   1.72949269e-01,


     8.87169778e-01,   1.12361245e-01,   5.84616959e-01,  -2.29443505e-01,


    -1.21431826e-02,  -4.25796807e-02,  -6.10006988e-01,  -6.49658367e-02,


     4.44499925e-02,   2.29505524e-01,   1.28365666e-01,   8.14887732e-02],


  [  7.38480210e-01,   1.66777715e-01,  -1.25522435e+00,  -9.91380811e-02,


    -7.94091165e-01,  -1.08379133e-01,  -2.26139322e-01,   1.61183283e-01,


     5.28308988e-01,   2.12008148e-01,   8.32218587e-01,   1.00101568e-01,


    -9.88249928e-02,  -1.80823252e-01,  -5.93611419e-01,  -1.92114711e-01,


    -6.95996433e-02,   3.17081690e-01,  -1.36234081e+00,   1.95782676e-01,


     2.41567016e-01,  -1.27664492e-01,  -3.35734665e-01,   3.22716296e-01,


    -5.90085804e-01,   3.65092635e-01,  -9.04714391e-02,   8.47652107e-02,


    -3.05684768e-02,  -5.65977693e-01,   3.81316900e-01,   1.04605407e-01,


     2.21714765e-01,   2.80387878e-01,  -9.49988514e-02,  -2.87344754e-01,


    -2.68851295e-02,   2.86572754e-01,  -3.87358405e-02,  -1.22303712e+00,


     2.79854000e-01,   6.24693893e-02,  -6.05009962e-03,  -2.06866369e-01,


     1.33305062e-02,  -5.31045258e-01,   1.14393413e-01,  -3.71956557e-01,


    -4.76430543e-02,  -2.57138878e-01,  -3.58281076e-01,   1.82284683e-01,


    -9.57218111e-02,   2.56797731e-01,  -2.66193479e-01,   2.66302407e-01,


    -1.47495762e-01,  -1.31557316e-01,  -1.64534017e-01,  -3.24462235e-01,


    -4.29839879e-01,  -1.34706739e-02,   2.29521975e-01,   2.86956340e-01]]]]




#layer1=l.ConvLayer(452,4,2,0,3,1,weights1)
layer1=l.ConvLayer(32,3,1,0,3,64,weights1,biases1)
#layer2=l.ConvLayer(225,2,5,0,1,1,weights2,biases1)
#im=Image.open("cat_origin.jpeg")
im=Image.open("Globe.png")
pic = np.array(im)
height=len(pic)
width=len(pic[0])
volume=[[[0 for i in range (width)] for j in range (height)] for k in range (3)]
for i in range(height):
    for j in range(width):
        for k in range(3):
            volume[k][i][j]=pic[i][j][k]
volume2=layer1.forward(volume)
pic2_0=np.array(volume2[0]).astype(np.uint8)
pic2_1=np.array(volume2[1]).astype(np.uint8)
im2_0=Image.fromarray(pic2_0)
im2_1=Image.fromarray(pic2_1)
#im2.convert('RGB')
im2_0.save("globe_int_0.png")
im2_1.save("globe_int_1.png")
#volume3=layer2.forward(volume2)
#pic3=np.array(volume3[0]).astype(np.uint8)
#im3=Image.fromarray(pic3)
#im3.save("cat_final_0.png")
