MyoSuite
========

EnvPool's MyoSuite integration uses ``myosuite==2.11.6`` pinned at commit
``05cb84678373f91271004f99602ebbf01e57d1a1`` with ``mujoco==3.6.0``.
The runtime implementation is native C++; the official Python package is used
only by oracle tests and doc-generation tooling.

The generated upstream registry and task metadata live under
``third_party/myosuite/``. Runtime C++ consumes those generated assets instead
of keeping a handwritten task list in ``envpool/mujoco/myosuite/``.


Env IDs
-------

EnvPool registers all 398 official MyoSuite task IDs from the pinned oracle.
Every official ID also has an EnvPool alias of the form
``MyoSuite/<official-id>``, for example:

::

  envpool.make_gymnasium("myoFingerReachFixed-v0")
  envpool.make_gymnasium("MyoSuite/myoFingerReachFixed-v0")

The full registered official IDs and EnvPool aliases are:

::

  Official ID                         EnvPool alias
  -----------                         -------------
  MyoHandAirplaneFixed-v0             MyoSuite/MyoHandAirplaneFixed-v0
  MyoHandAirplaneFly-v0               MyoSuite/MyoHandAirplaneFly-v0
  MyoHandAirplaneLift-v0              MyoSuite/MyoHandAirplaneLift-v0
  MyoHandAirplanePass-v0              MyoSuite/MyoHandAirplanePass-v0
  MyoHandAirplaneRandom-v0            MyoSuite/MyoHandAirplaneRandom-v0
  MyoHandAlarmclockFixed-v0           MyoSuite/MyoHandAlarmclockFixed-v0
  MyoHandAlarmclockLift-v0            MyoSuite/MyoHandAlarmclockLift-v0
  MyoHandAlarmclockPass-v0            MyoSuite/MyoHandAlarmclockPass-v0
  MyoHandAlarmclockRandom-v0          MyoSuite/MyoHandAlarmclockRandom-v0
  MyoHandAlarmclockSee-v0             MyoSuite/MyoHandAlarmclockSee-v0
  MyoHandAppleFixed-v0                MyoSuite/MyoHandAppleFixed-v0
  MyoHandAppleLift-v0                 MyoSuite/MyoHandAppleLift-v0
  MyoHandApplePass-v0                 MyoSuite/MyoHandApplePass-v0
  MyoHandAppleRandom-v0               MyoSuite/MyoHandAppleRandom-v0
  MyoHandBananaFixed-v0               MyoSuite/MyoHandBananaFixed-v0
  MyoHandBananaPass-v0                MyoSuite/MyoHandBananaPass-v0
  MyoHandBananaRandom-v0              MyoSuite/MyoHandBananaRandom-v0
  MyoHandBinocularsFixed-v0           MyoSuite/MyoHandBinocularsFixed-v0
  MyoHandBinocularsPass-v0            MyoSuite/MyoHandBinocularsPass-v0
  MyoHandBinocularsRandom-v0          MyoSuite/MyoHandBinocularsRandom-v0
  MyoHandBowlDrink2-v0                MyoSuite/MyoHandBowlDrink2-v0
  MyoHandBowlFixed-v0                 MyoSuite/MyoHandBowlFixed-v0
  MyoHandBowlPass-v0                  MyoSuite/MyoHandBowlPass-v0
  MyoHandBowlRandom-v0                MyoSuite/MyoHandBowlRandom-v0
  MyoHandCameraFixed-v0               MyoSuite/MyoHandCameraFixed-v0
  MyoHandCameraPass-v0                MyoSuite/MyoHandCameraPass-v0
  MyoHandCameraRandom-v0              MyoSuite/MyoHandCameraRandom-v0
  MyoHandCoffeemugFixed-v0            MyoSuite/MyoHandCoffeemugFixed-v0
  MyoHandCoffeemugRandom-v0           MyoSuite/MyoHandCoffeemugRandom-v0
  MyoHandCubelargeFixed-v0            MyoSuite/MyoHandCubelargeFixed-v0
  MyoHandCubelargePass-v0             MyoSuite/MyoHandCubelargePass-v0
  MyoHandCubelargeRandom-v0           MyoSuite/MyoHandCubelargeRandom-v0
  MyoHandCubemediumFixed-v0           MyoSuite/MyoHandCubemediumFixed-v0
  MyoHandCubemediumLInspect-v0        MyoSuite/MyoHandCubemediumLInspect-v0
  MyoHandCubemediumRandom-v0          MyoSuite/MyoHandCubemediumRandom-v0
  MyoHandCubesmallFixed-v0            MyoSuite/MyoHandCubesmallFixed-v0
  MyoHandCubesmallLift-v0             MyoSuite/MyoHandCubesmallLift-v0
  MyoHandCubesmallPass-v0             MyoSuite/MyoHandCubesmallPass-v0
  MyoHandCubesmallRandom-v0           MyoSuite/MyoHandCubesmallRandom-v0
  MyoHandCupDrink-v0                  MyoSuite/MyoHandCupDrink-v0
  MyoHandCupFixed-v0                  MyoSuite/MyoHandCupFixed-v0
  MyoHandCupPass-v0                   MyoSuite/MyoHandCupPass-v0
  MyoHandCupPour-v0                   MyoSuite/MyoHandCupPour-v0
  MyoHandCupRandom-v0                 MyoSuite/MyoHandCupRandom-v0
  MyoHandCylinderlargeFixed-v0        MyoSuite/MyoHandCylinderlargeFixed-v0
  MyoHandCylinderlargeInspect-v0      MyoSuite/MyoHandCylinderlargeInspect-v0
  MyoHandCylinderlargeRandom-v0       MyoSuite/MyoHandCylinderlargeRandom-v0
  MyoHandCylindermediumFixed-v0       MyoSuite/MyoHandCylindermediumFixed-v0
  MyoHandCylindermediumLift-v0        MyoSuite/MyoHandCylindermediumLift-v0
  MyoHandCylindermediumPass-v0        MyoSuite/MyoHandCylindermediumPass-v0
  MyoHandCylindermediumRandom-v0      MyoSuite/MyoHandCylindermediumRandom-v0
  MyoHandCylindersmallFixed-v0        MyoSuite/MyoHandCylindersmallFixed-v0
  MyoHandCylindersmallInspect-v0      MyoSuite/MyoHandCylindersmallInspect-v0
  MyoHandCylindersmallPass-v0         MyoSuite/MyoHandCylindersmallPass-v0
  MyoHandCylindersmallRandom-v0       MyoSuite/MyoHandCylindersmallRandom-v0
  MyoHandDuckFixed-v0                 MyoSuite/MyoHandDuckFixed-v0
  MyoHandDuckInspect-v0               MyoSuite/MyoHandDuckInspect-v0
  MyoHandDuckLift-v0                  MyoSuite/MyoHandDuckLift-v0
  MyoHandDuckPass-v0                  MyoSuite/MyoHandDuckPass-v0
  MyoHandDuckRandom-v0                MyoSuite/MyoHandDuckRandom-v0
  MyoHandElephantFixed-v0             MyoSuite/MyoHandElephantFixed-v0
  MyoHandElephantLift-v0              MyoSuite/MyoHandElephantLift-v0
  MyoHandElephantPass-v0              MyoSuite/MyoHandElephantPass-v0
  MyoHandElephantRandom-v0            MyoSuite/MyoHandElephantRandom-v0
  MyoHandEyeglassesFixed-v0           MyoSuite/MyoHandEyeglassesFixed-v0
  MyoHandEyeglassesPass-v0            MyoSuite/MyoHandEyeglassesPass-v0
  MyoHandEyeglassesRandom-v0          MyoSuite/MyoHandEyeglassesRandom-v0
  MyoHandFlashlight1On-v0             MyoSuite/MyoHandFlashlight1On-v0
  MyoHandFlashlight2On-v0             MyoSuite/MyoHandFlashlight2On-v0
  MyoHandFlashlightFixed-v0           MyoSuite/MyoHandFlashlightFixed-v0
  MyoHandFlashlightLift-v0            MyoSuite/MyoHandFlashlightLift-v0
  MyoHandFlashlightPass-v0            MyoSuite/MyoHandFlashlightPass-v0
  MyoHandFlashlightRandom-v0          MyoSuite/MyoHandFlashlightRandom-v0
  MyoHandFluteFixed-v0                MyoSuite/MyoHandFluteFixed-v0
  MyoHandFlutePass-v0                 MyoSuite/MyoHandFlutePass-v0
  MyoHandFluteRandom-v0               MyoSuite/MyoHandFluteRandom-v0
  MyoHandGamecontrollerFixed-v0       MyoSuite/MyoHandGamecontrollerFixed-v0
  MyoHandGamecontrollerPass-v0        MyoSuite/MyoHandGamecontrollerPass-v0
  MyoHandGamecontrollerRandom-v0      MyoSuite/MyoHandGamecontrollerRandom-v0
  MyoHandHammerFixed-v0               MyoSuite/MyoHandHammerFixed-v0
  MyoHandHammerPass-v0                MyoSuite/MyoHandHammerPass-v0
  MyoHandHammerRandom-v0              MyoSuite/MyoHandHammerRandom-v0
  MyoHandHammerUse-v0                 MyoSuite/MyoHandHammerUse-v0
  MyoHandHandFixed-v0                 MyoSuite/MyoHandHandFixed-v0
  MyoHandHandInspect-v0               MyoSuite/MyoHandHandInspect-v0
  MyoHandHandRandom-v0                MyoSuite/MyoHandHandRandom-v0
  MyoHandHeadphonesFixed-v0           MyoSuite/MyoHandHeadphonesFixed-v0
  MyoHandHeadphonesPass-v0            MyoSuite/MyoHandHeadphonesPass-v0
  MyoHandHeadphonesRandom-v0          MyoSuite/MyoHandHeadphonesRandom-v0
  MyoHandKnifeChop-v0                 MyoSuite/MyoHandKnifeChop-v0
  MyoHandKnifeFixed-v0                MyoSuite/MyoHandKnifeFixed-v0
  MyoHandKnifeRandom-v0               MyoSuite/MyoHandKnifeRandom-v0
  MyoHandLightbulbFixed-v0            MyoSuite/MyoHandLightbulbFixed-v0
  MyoHandLightbulbPass-v0             MyoSuite/MyoHandLightbulbPass-v0
  MyoHandLightbulbRandom-v0           MyoSuite/MyoHandLightbulbRandom-v0
  MyoHandMouseFixed-v0                MyoSuite/MyoHandMouseFixed-v0
  MyoHandMouseLift-v0                 MyoSuite/MyoHandMouseLift-v0
  MyoHandMousePass-v0                 MyoSuite/MyoHandMousePass-v0
  MyoHandMouseRandom-v0               MyoSuite/MyoHandMouseRandom-v0
  MyoHandMouseUse-v0                  MyoSuite/MyoHandMouseUse-v0
  MyoHandMugDrink3-v0                 MyoSuite/MyoHandMugDrink3-v0
  MyoHandMugFixed-v0                  MyoSuite/MyoHandMugFixed-v0
  MyoHandMugLift-v0                   MyoSuite/MyoHandMugLift-v0
  MyoHandMugPass-v0                   MyoSuite/MyoHandMugPass-v0
  MyoHandMugRandom-v0                 MyoSuite/MyoHandMugRandom-v0
  MyoHandPhoneFixed-v0                MyoSuite/MyoHandPhoneFixed-v0
  MyoHandPhoneLift-v0                 MyoSuite/MyoHandPhoneLift-v0
  MyoHandPhoneRandom-v0               MyoSuite/MyoHandPhoneRandom-v0
  MyoHandPiggybankFixed-v0            MyoSuite/MyoHandPiggybankFixed-v0
  MyoHandPiggybankPass-v0             MyoSuite/MyoHandPiggybankPass-v0
  MyoHandPiggybankRandom-v0           MyoSuite/MyoHandPiggybankRandom-v0
  MyoHandPiggybankUse-v0              MyoSuite/MyoHandPiggybankUse-v0
  MyoHandPyramidlargeFixed-v0         MyoSuite/MyoHandPyramidlargeFixed-v0
  MyoHandPyramidlargePass-v0          MyoSuite/MyoHandPyramidlargePass-v0
  MyoHandPyramidlargeRandom-v0        MyoSuite/MyoHandPyramidlargeRandom-v0
  MyoHandPyramidmediumFixed-v0        MyoSuite/MyoHandPyramidmediumFixed-v0
  MyoHandPyramidmediumPass-v0         MyoSuite/MyoHandPyramidmediumPass-v0
  MyoHandPyramidmediumRandom-v0       MyoSuite/MyoHandPyramidmediumRandom-v0
  MyoHandPyramidsmallFixed-v0         MyoSuite/MyoHandPyramidsmallFixed-v0
  MyoHandPyramidsmallInspect-v0       MyoSuite/MyoHandPyramidsmallInspect-v0
  MyoHandPyramidsmallRandom-v0        MyoSuite/MyoHandPyramidsmallRandom-v0
  MyoHandScissorsFixed-v0             MyoSuite/MyoHandScissorsFixed-v0
  MyoHandScissorsRandom-v0            MyoSuite/MyoHandScissorsRandom-v0
  MyoHandScissorsUse-v0               MyoSuite/MyoHandScissorsUse-v0
  MyoHandSpherelargeFixed-v0          MyoSuite/MyoHandSpherelargeFixed-v0
  MyoHandSpherelargePass-v0           MyoSuite/MyoHandSpherelargePass-v0
  MyoHandSpherelargeRandom-v0         MyoSuite/MyoHandSpherelargeRandom-v0
  MyoHandSpheremediumFixed-v0         MyoSuite/MyoHandSpheremediumFixed-v0
  MyoHandSpheremediumInspect-v0       MyoSuite/MyoHandSpheremediumInspect-v0
  MyoHandSpheremediumLift-v0          MyoSuite/MyoHandSpheremediumLift-v0
  MyoHandSpheremediumRandom-v0        MyoSuite/MyoHandSpheremediumRandom-v0
  MyoHandSpheresmallFixed-v0          MyoSuite/MyoHandSpheresmallFixed-v0
  MyoHandSpheresmallInspect-v0        MyoSuite/MyoHandSpheresmallInspect-v0
  MyoHandSpheresmallLift-v0           MyoSuite/MyoHandSpheresmallLift-v0
  MyoHandSpheresmallPass-v0           MyoSuite/MyoHandSpheresmallPass-v0
  MyoHandSpheresmallRandom-v0         MyoSuite/MyoHandSpheresmallRandom-v0
  MyoHandStampFixed-v0                MyoSuite/MyoHandStampFixed-v0
  MyoHandStampLift-v0                 MyoSuite/MyoHandStampLift-v0
  MyoHandStampRandom-v0               MyoSuite/MyoHandStampRandom-v0
  MyoHandStampStamp-v0                MyoSuite/MyoHandStampStamp-v0
  MyoHandStanfordbunnyFixed-v0        MyoSuite/MyoHandStanfordbunnyFixed-v0
  MyoHandStanfordbunnyInspect-v0      MyoSuite/MyoHandStanfordbunnyInspect-v0
  MyoHandStanfordbunnyPass-v0         MyoSuite/MyoHandStanfordbunnyPass-v0
  MyoHandStanfordbunnyRandom-v0       MyoSuite/MyoHandStanfordbunnyRandom-v0
  MyoHandStaplerFixed-v0              MyoSuite/MyoHandStaplerFixed-v0
  MyoHandStaplerLift-v0               MyoSuite/MyoHandStaplerLift-v0
  MyoHandStaplerRandom-v0             MyoSuite/MyoHandStaplerRandom-v0
  MyoHandStaplerStaple1-v0            MyoSuite/MyoHandStaplerStaple1-v0
  MyoHandStaplerStaple2-v0            MyoSuite/MyoHandStaplerStaple2-v0
  MyoHandTeapotFixed-v0               MyoSuite/MyoHandTeapotFixed-v0
  MyoHandTeapotPour2-v0               MyoSuite/MyoHandTeapotPour2-v0
  MyoHandTeapotRandom-v0              MyoSuite/MyoHandTeapotRandom-v0
  MyoHandToothbrushBrush1-v0          MyoSuite/MyoHandToothbrushBrush1-v0
  MyoHandToothbrushFixed-v0           MyoSuite/MyoHandToothbrushFixed-v0
  MyoHandToothbrushLift-v0            MyoSuite/MyoHandToothbrushLift-v0
  MyoHandToothbrushRandom-v0          MyoSuite/MyoHandToothbrushRandom-v0
  MyoHandToothpasteFixed-v0           MyoSuite/MyoHandToothpasteFixed-v0
  MyoHandToothpasteLift-v0            MyoSuite/MyoHandToothpasteLift-v0
  MyoHandToothpasteRandom-v0          MyoSuite/MyoHandToothpasteRandom-v0
  MyoHandToothpasteSqueeze1-v0        MyoSuite/MyoHandToothpasteSqueeze1-v0
  MyoHandToruslargeFixed-v0           MyoSuite/MyoHandToruslargeFixed-v0
  MyoHandToruslargeInspect-v0         MyoSuite/MyoHandToruslargeInspect-v0
  MyoHandToruslargeLift-v0            MyoSuite/MyoHandToruslargeLift-v0
  MyoHandToruslargeRandom-v0          MyoSuite/MyoHandToruslargeRandom-v0
  MyoHandTorusmediumFixed-v0          MyoSuite/MyoHandTorusmediumFixed-v0
  MyoHandTorusmediumLift-v0           MyoSuite/MyoHandTorusmediumLift-v0
  MyoHandTorusmediumPass-v0           MyoSuite/MyoHandTorusmediumPass-v0
  MyoHandTorusmediumRandom-v0         MyoSuite/MyoHandTorusmediumRandom-v0
  MyoHandTorussmallFixed-v0           MyoSuite/MyoHandTorussmallFixed-v0
  MyoHandTorussmallLift-v0            MyoSuite/MyoHandTorussmallLift-v0
  MyoHandTorussmallPass-v0            MyoSuite/MyoHandTorussmallPass-v0
  MyoHandTorussmallRandom-v0          MyoSuite/MyoHandTorussmallRandom-v0
  MyoHandTrainFixed-v0                MyoSuite/MyoHandTrainFixed-v0
  MyoHandTrainPlay-v0                 MyoSuite/MyoHandTrainPlay-v0
  MyoHandTrainRandom-v0               MyoSuite/MyoHandTrainRandom-v0
  MyoHandWatchFixed-v0                MyoSuite/MyoHandWatchFixed-v0
  MyoHandWatchLift-v0                 MyoSuite/MyoHandWatchLift-v0
  MyoHandWatchPass-v0                 MyoSuite/MyoHandWatchPass-v0
  MyoHandWatchRandom-v0               MyoSuite/MyoHandWatchRandom-v0
  MyoHandWaterbottleFixed-v0          MyoSuite/MyoHandWaterbottleFixed-v0
  MyoHandWaterbottleLift-v0           MyoSuite/MyoHandWaterbottleLift-v0
  MyoHandWaterbottlePass-v0           MyoSuite/MyoHandWaterbottlePass-v0
  MyoHandWaterbottleRandom-v0         MyoSuite/MyoHandWaterbottleRandom-v0
  MyoHandWaterbottleShake-v0          MyoSuite/MyoHandWaterbottleShake-v0
  MyoHandWineglassDrink2-v0           MyoSuite/MyoHandWineglassDrink2-v0
  MyoHandWineglassFixed-v0            MyoSuite/MyoHandWineglassFixed-v0
  MyoHandWineglassLift-v0             MyoSuite/MyoHandWineglassLift-v0
  MyoHandWineglassPass-v0             MyoSuite/MyoHandWineglassPass-v0
  MyoHandWineglassRandom-v0           MyoSuite/MyoHandWineglassRandom-v0
  MyoHandWineglassToast1-v0           MyoSuite/MyoHandWineglassToast1-v0
  motorFingerPoseFixed-v0             MyoSuite/motorFingerPoseFixed-v0
  motorFingerPoseRandom-v0            MyoSuite/motorFingerPoseRandom-v0
  motorFingerReachFixed-v0            MyoSuite/motorFingerReachFixed-v0
  motorFingerReachRandom-v0           MyoSuite/motorFingerReachRandom-v0
  myoArmReachFixed-v0                 MyoSuite/myoArmReachFixed-v0
  myoArmReachRandom-v0                MyoSuite/myoArmReachRandom-v0
  myoChallengeBaodingP1-v1            MyoSuite/myoChallengeBaodingP1-v1
  myoChallengeBaodingP2-v1            MyoSuite/myoChallengeBaodingP2-v1
  myoChallengeBimanual-v0             MyoSuite/myoChallengeBimanual-v0
  myoChallengeChaseTagP1-v0           MyoSuite/myoChallengeChaseTagP1-v0
  myoChallengeChaseTagP2-v0           MyoSuite/myoChallengeChaseTagP2-v0
  myoChallengeChaseTagP2eval-v0       MyoSuite/myoChallengeChaseTagP2eval-v0
  myoChallengeDieReorientDemo-v0      MyoSuite/myoChallengeDieReorientDemo-v0
  myoChallengeDieReorientP1-v0        MyoSuite/myoChallengeDieReorientP1-v0
  myoChallengeDieReorientP2-v0        MyoSuite/myoChallengeDieReorientP2-v0
  myoChallengeOslRunFixed-v0          MyoSuite/myoChallengeOslRunFixed-v0
  myoChallengeOslRunRandom-v0         MyoSuite/myoChallengeOslRunRandom-v0
  myoChallengeRelocateP1-v0           MyoSuite/myoChallengeRelocateP1-v0
  myoChallengeRelocateP2-v0           MyoSuite/myoChallengeRelocateP2-v0
  myoChallengeRelocateP2eval-v0       MyoSuite/myoChallengeRelocateP2eval-v0
  myoChallengeSoccerP1-v0             MyoSuite/myoChallengeSoccerP1-v0
  myoChallengeSoccerP2-v0             MyoSuite/myoChallengeSoccerP2-v0
  myoChallengeTableTennisP0-v0        MyoSuite/myoChallengeTableTennisP0-v0
  myoChallengeTableTennisP1-v0        MyoSuite/myoChallengeTableTennisP1-v0
  myoChallengeTableTennisP2-v0        MyoSuite/myoChallengeTableTennisP2-v0
  myoElbowPose1D6MExoFixed-v0         MyoSuite/myoElbowPose1D6MExoFixed-v0
  myoElbowPose1D6MExoRandom-v0        MyoSuite/myoElbowPose1D6MExoRandom-v0
  myoElbowPose1D6MFixed-v0            MyoSuite/myoElbowPose1D6MFixed-v0
  myoElbowPose1D6MRandom-v0           MyoSuite/myoElbowPose1D6MRandom-v0
  myoFatiArmReachFixed-v0             MyoSuite/myoFatiArmReachFixed-v0
  myoFatiArmReachRandom-v0            MyoSuite/myoFatiArmReachRandom-v0
  myoFatiChallengeBaodingP1-v1        MyoSuite/myoFatiChallengeBaodingP1-v1
  myoFatiChallengeBaodingP2-v1        MyoSuite/myoFatiChallengeBaodingP2-v1
  myoFatiChallengeBimanual-v0         MyoSuite/myoFatiChallengeBimanual-v0
  myoFatiChallengeChaseTagP1-v0       MyoSuite/myoFatiChallengeChaseTagP1-v0
  myoFatiChallengeChaseTagP2-v0       MyoSuite/myoFatiChallengeChaseTagP2-v0
  myoFatiChallengeChaseTagP2eval-v0   MyoSuite/myoFatiChallengeChaseTagP2eval-v0
  myoFatiChallengeDieReorientDemo-v0  MyoSuite/myoFatiChallengeDieReorientDemo-v0
  myoFatiChallengeDieReorientP1-v0    MyoSuite/myoFatiChallengeDieReorientP1-v0
  myoFatiChallengeDieReorientP2-v0    MyoSuite/myoFatiChallengeDieReorientP2-v0
  myoFatiChallengeOslRunFixed-v0      MyoSuite/myoFatiChallengeOslRunFixed-v0
  myoFatiChallengeOslRunRandom-v0     MyoSuite/myoFatiChallengeOslRunRandom-v0
  myoFatiChallengeRelocateP1-v0       MyoSuite/myoFatiChallengeRelocateP1-v0
  myoFatiChallengeRelocateP2-v0       MyoSuite/myoFatiChallengeRelocateP2-v0
  myoFatiChallengeRelocateP2eval-v0   MyoSuite/myoFatiChallengeRelocateP2eval-v0
  myoFatiChallengeSoccerP1-v0         MyoSuite/myoFatiChallengeSoccerP1-v0
  myoFatiChallengeSoccerP2-v0         MyoSuite/myoFatiChallengeSoccerP2-v0
  myoFatiChallengeTableTennisP0-v0    MyoSuite/myoFatiChallengeTableTennisP0-v0
  myoFatiChallengeTableTennisP1-v0    MyoSuite/myoFatiChallengeTableTennisP1-v0
  myoFatiChallengeTableTennisP2-v0    MyoSuite/myoFatiChallengeTableTennisP2-v0
  myoFatiElbowPose1D6MExoFixed-v0     MyoSuite/myoFatiElbowPose1D6MExoFixed-v0
  myoFatiElbowPose1D6MExoRandom-v0    MyoSuite/myoFatiElbowPose1D6MExoRandom-v0
  myoFatiElbowPose1D6MFixed-v0        MyoSuite/myoFatiElbowPose1D6MFixed-v0
  myoFatiElbowPose1D6MRandom-v0       MyoSuite/myoFatiElbowPose1D6MRandom-v0
  myoFatiFingerPoseFixed-v0           MyoSuite/myoFatiFingerPoseFixed-v0
  myoFatiFingerPoseRandom-v0          MyoSuite/myoFatiFingerPoseRandom-v0
  myoFatiFingerReachFixed-v0          MyoSuite/myoFatiFingerReachFixed-v0
  myoFatiFingerReachRandom-v0         MyoSuite/myoFatiFingerReachRandom-v0
  myoFatiHandKeyTurnFixed-v0          MyoSuite/myoFatiHandKeyTurnFixed-v0
  myoFatiHandKeyTurnRandom-v0         MyoSuite/myoFatiHandKeyTurnRandom-v0
  myoFatiHandObjHoldFixed-v0          MyoSuite/myoFatiHandObjHoldFixed-v0
  myoFatiHandObjHoldRandom-v0         MyoSuite/myoFatiHandObjHoldRandom-v0
  myoFatiHandPenTwirlFixed-v0         MyoSuite/myoFatiHandPenTwirlFixed-v0
  myoFatiHandPenTwirlRandom-v0        MyoSuite/myoFatiHandPenTwirlRandom-v0
  myoFatiHandPose0Fixed-v0            MyoSuite/myoFatiHandPose0Fixed-v0
  myoFatiHandPose1Fixed-v0            MyoSuite/myoFatiHandPose1Fixed-v0
  myoFatiHandPose2Fixed-v0            MyoSuite/myoFatiHandPose2Fixed-v0
  myoFatiHandPose3Fixed-v0            MyoSuite/myoFatiHandPose3Fixed-v0
  myoFatiHandPose4Fixed-v0            MyoSuite/myoFatiHandPose4Fixed-v0
  myoFatiHandPose5Fixed-v0            MyoSuite/myoFatiHandPose5Fixed-v0
  myoFatiHandPose6Fixed-v0            MyoSuite/myoFatiHandPose6Fixed-v0
  myoFatiHandPose7Fixed-v0            MyoSuite/myoFatiHandPose7Fixed-v0
  myoFatiHandPose8Fixed-v0            MyoSuite/myoFatiHandPose8Fixed-v0
  myoFatiHandPose9Fixed-v0            MyoSuite/myoFatiHandPose9Fixed-v0
  myoFatiHandPoseFixed-v0             MyoSuite/myoFatiHandPoseFixed-v0
  myoFatiHandPoseRandom-v0            MyoSuite/myoFatiHandPoseRandom-v0
  myoFatiHandReachFixed-v0            MyoSuite/myoFatiHandReachFixed-v0
  myoFatiHandReachRandom-v0           MyoSuite/myoFatiHandReachRandom-v0
  myoFatiHandReorient100-v0           MyoSuite/myoFatiHandReorient100-v0
  myoFatiHandReorient8-v0             MyoSuite/myoFatiHandReorient8-v0
  myoFatiHandReorientID-v0            MyoSuite/myoFatiHandReorientID-v0
  myoFatiHandReorientOOD-v0           MyoSuite/myoFatiHandReorientOOD-v0
  myoFatiLegHillyTerrainWalk-v0       MyoSuite/myoFatiLegHillyTerrainWalk-v0
  myoFatiLegRoughTerrainWalk-v0       MyoSuite/myoFatiLegRoughTerrainWalk-v0
  myoFatiLegStairTerrainWalk-v0       MyoSuite/myoFatiLegStairTerrainWalk-v0
  myoFatiLegStandRandom-v0            MyoSuite/myoFatiLegStandRandom-v0
  myoFatiLegWalk-v0                   MyoSuite/myoFatiLegWalk-v0
  myoFatiTorsoExoPoseFixed-v0         MyoSuite/myoFatiTorsoExoPoseFixed-v0
  myoFatiTorsoPoseFixed-v0            MyoSuite/myoFatiTorsoPoseFixed-v0
  myoFingerPoseFixed-v0               MyoSuite/myoFingerPoseFixed-v0
  myoFingerPoseRandom-v0              MyoSuite/myoFingerPoseRandom-v0
  myoFingerReachFixed-v0              MyoSuite/myoFingerReachFixed-v0
  myoFingerReachRandom-v0             MyoSuite/myoFingerReachRandom-v0
  myoHandKeyTurnFixed-v0              MyoSuite/myoHandKeyTurnFixed-v0
  myoHandKeyTurnRandom-v0             MyoSuite/myoHandKeyTurnRandom-v0
  myoHandObjHoldFixed-v0              MyoSuite/myoHandObjHoldFixed-v0
  myoHandObjHoldRandom-v0             MyoSuite/myoHandObjHoldRandom-v0
  myoHandPenTwirlFixed-v0             MyoSuite/myoHandPenTwirlFixed-v0
  myoHandPenTwirlRandom-v0            MyoSuite/myoHandPenTwirlRandom-v0
  myoHandPose0Fixed-v0                MyoSuite/myoHandPose0Fixed-v0
  myoHandPose1Fixed-v0                MyoSuite/myoHandPose1Fixed-v0
  myoHandPose2Fixed-v0                MyoSuite/myoHandPose2Fixed-v0
  myoHandPose3Fixed-v0                MyoSuite/myoHandPose3Fixed-v0
  myoHandPose4Fixed-v0                MyoSuite/myoHandPose4Fixed-v0
  myoHandPose5Fixed-v0                MyoSuite/myoHandPose5Fixed-v0
  myoHandPose6Fixed-v0                MyoSuite/myoHandPose6Fixed-v0
  myoHandPose7Fixed-v0                MyoSuite/myoHandPose7Fixed-v0
  myoHandPose8Fixed-v0                MyoSuite/myoHandPose8Fixed-v0
  myoHandPose9Fixed-v0                MyoSuite/myoHandPose9Fixed-v0
  myoHandPoseFixed-v0                 MyoSuite/myoHandPoseFixed-v0
  myoHandPoseRandom-v0                MyoSuite/myoHandPoseRandom-v0
  myoHandReachFixed-v0                MyoSuite/myoHandReachFixed-v0
  myoHandReachRandom-v0               MyoSuite/myoHandReachRandom-v0
  myoHandReorient100-v0               MyoSuite/myoHandReorient100-v0
  myoHandReorient8-v0                 MyoSuite/myoHandReorient8-v0
  myoHandReorientID-v0                MyoSuite/myoHandReorientID-v0
  myoHandReorientOOD-v0               MyoSuite/myoHandReorientOOD-v0
  myoLegHillyTerrainWalk-v0           MyoSuite/myoLegHillyTerrainWalk-v0
  myoLegRoughTerrainWalk-v0           MyoSuite/myoLegRoughTerrainWalk-v0
  myoLegStairTerrainWalk-v0           MyoSuite/myoLegStairTerrainWalk-v0
  myoLegStandRandom-v0                MyoSuite/myoLegStandRandom-v0
  myoLegWalk-v0                       MyoSuite/myoLegWalk-v0
  myoReafHandKeyTurnFixed-v0          MyoSuite/myoReafHandKeyTurnFixed-v0
  myoReafHandKeyTurnRandom-v0         MyoSuite/myoReafHandKeyTurnRandom-v0
  myoReafHandObjHoldFixed-v0          MyoSuite/myoReafHandObjHoldFixed-v0
  myoReafHandObjHoldRandom-v0         MyoSuite/myoReafHandObjHoldRandom-v0
  myoReafHandPenTwirlFixed-v0         MyoSuite/myoReafHandPenTwirlFixed-v0
  myoReafHandPenTwirlRandom-v0        MyoSuite/myoReafHandPenTwirlRandom-v0
  myoReafHandPose0Fixed-v0            MyoSuite/myoReafHandPose0Fixed-v0
  myoReafHandPose1Fixed-v0            MyoSuite/myoReafHandPose1Fixed-v0
  myoReafHandPose2Fixed-v0            MyoSuite/myoReafHandPose2Fixed-v0
  myoReafHandPose3Fixed-v0            MyoSuite/myoReafHandPose3Fixed-v0
  myoReafHandPose4Fixed-v0            MyoSuite/myoReafHandPose4Fixed-v0
  myoReafHandPose5Fixed-v0            MyoSuite/myoReafHandPose5Fixed-v0
  myoReafHandPose6Fixed-v0            MyoSuite/myoReafHandPose6Fixed-v0
  myoReafHandPose7Fixed-v0            MyoSuite/myoReafHandPose7Fixed-v0
  myoReafHandPose8Fixed-v0            MyoSuite/myoReafHandPose8Fixed-v0
  myoReafHandPose9Fixed-v0            MyoSuite/myoReafHandPose9Fixed-v0
  myoReafHandPoseFixed-v0             MyoSuite/myoReafHandPoseFixed-v0
  myoReafHandPoseRandom-v0            MyoSuite/myoReafHandPoseRandom-v0
  myoReafHandReachFixed-v0            MyoSuite/myoReafHandReachFixed-v0
  myoReafHandReachRandom-v0           MyoSuite/myoReafHandReachRandom-v0
  myoReafHandReorient100-v0           MyoSuite/myoReafHandReorient100-v0
  myoReafHandReorient8-v0             MyoSuite/myoReafHandReorient8-v0
  myoReafHandReorientID-v0            MyoSuite/myoReafHandReorientID-v0
  myoReafHandReorientOOD-v0           MyoSuite/myoReafHandReorientOOD-v0
  myoSarcArmReachFixed-v0             MyoSuite/myoSarcArmReachFixed-v0
  myoSarcArmReachRandom-v0            MyoSuite/myoSarcArmReachRandom-v0
  myoSarcChallengeBaodingP1-v1        MyoSuite/myoSarcChallengeBaodingP1-v1
  myoSarcChallengeBaodingP2-v1        MyoSuite/myoSarcChallengeBaodingP2-v1
  myoSarcChallengeBimanual-v0         MyoSuite/myoSarcChallengeBimanual-v0
  myoSarcChallengeChaseTagP1-v0       MyoSuite/myoSarcChallengeChaseTagP1-v0
  myoSarcChallengeChaseTagP2-v0       MyoSuite/myoSarcChallengeChaseTagP2-v0
  myoSarcChallengeChaseTagP2eval-v0   MyoSuite/myoSarcChallengeChaseTagP2eval-v0
  myoSarcChallengeDieReorientDemo-v0  MyoSuite/myoSarcChallengeDieReorientDemo-v0
  myoSarcChallengeDieReorientP1-v0    MyoSuite/myoSarcChallengeDieReorientP1-v0
  myoSarcChallengeDieReorientP2-v0    MyoSuite/myoSarcChallengeDieReorientP2-v0
  myoSarcChallengeOslRunFixed-v0      MyoSuite/myoSarcChallengeOslRunFixed-v0
  myoSarcChallengeOslRunRandom-v0     MyoSuite/myoSarcChallengeOslRunRandom-v0
  myoSarcChallengeRelocateP1-v0       MyoSuite/myoSarcChallengeRelocateP1-v0
  myoSarcChallengeRelocateP2-v0       MyoSuite/myoSarcChallengeRelocateP2-v0
  myoSarcChallengeRelocateP2eval-v0   MyoSuite/myoSarcChallengeRelocateP2eval-v0
  myoSarcChallengeSoccerP1-v0         MyoSuite/myoSarcChallengeSoccerP1-v0
  myoSarcChallengeSoccerP2-v0         MyoSuite/myoSarcChallengeSoccerP2-v0
  myoSarcChallengeTableTennisP0-v0    MyoSuite/myoSarcChallengeTableTennisP0-v0
  myoSarcChallengeTableTennisP1-v0    MyoSuite/myoSarcChallengeTableTennisP1-v0
  myoSarcChallengeTableTennisP2-v0    MyoSuite/myoSarcChallengeTableTennisP2-v0
  myoSarcElbowPose1D6MExoFixed-v0     MyoSuite/myoSarcElbowPose1D6MExoFixed-v0
  myoSarcElbowPose1D6MExoRandom-v0    MyoSuite/myoSarcElbowPose1D6MExoRandom-v0
  myoSarcElbowPose1D6MFixed-v0        MyoSuite/myoSarcElbowPose1D6MFixed-v0
  myoSarcElbowPose1D6MRandom-v0       MyoSuite/myoSarcElbowPose1D6MRandom-v0
  myoSarcFingerPoseFixed-v0           MyoSuite/myoSarcFingerPoseFixed-v0
  myoSarcFingerPoseRandom-v0          MyoSuite/myoSarcFingerPoseRandom-v0
  myoSarcFingerReachFixed-v0          MyoSuite/myoSarcFingerReachFixed-v0
  myoSarcFingerReachRandom-v0         MyoSuite/myoSarcFingerReachRandom-v0
  myoSarcHandKeyTurnFixed-v0          MyoSuite/myoSarcHandKeyTurnFixed-v0
  myoSarcHandKeyTurnRandom-v0         MyoSuite/myoSarcHandKeyTurnRandom-v0
  myoSarcHandObjHoldFixed-v0          MyoSuite/myoSarcHandObjHoldFixed-v0
  myoSarcHandObjHoldRandom-v0         MyoSuite/myoSarcHandObjHoldRandom-v0
  myoSarcHandPenTwirlFixed-v0         MyoSuite/myoSarcHandPenTwirlFixed-v0
  myoSarcHandPenTwirlRandom-v0        MyoSuite/myoSarcHandPenTwirlRandom-v0
  myoSarcHandPose0Fixed-v0            MyoSuite/myoSarcHandPose0Fixed-v0
  myoSarcHandPose1Fixed-v0            MyoSuite/myoSarcHandPose1Fixed-v0
  myoSarcHandPose2Fixed-v0            MyoSuite/myoSarcHandPose2Fixed-v0
  myoSarcHandPose3Fixed-v0            MyoSuite/myoSarcHandPose3Fixed-v0
  myoSarcHandPose4Fixed-v0            MyoSuite/myoSarcHandPose4Fixed-v0
  myoSarcHandPose5Fixed-v0            MyoSuite/myoSarcHandPose5Fixed-v0
  myoSarcHandPose6Fixed-v0            MyoSuite/myoSarcHandPose6Fixed-v0
  myoSarcHandPose7Fixed-v0            MyoSuite/myoSarcHandPose7Fixed-v0
  myoSarcHandPose8Fixed-v0            MyoSuite/myoSarcHandPose8Fixed-v0
  myoSarcHandPose9Fixed-v0            MyoSuite/myoSarcHandPose9Fixed-v0
  myoSarcHandPoseFixed-v0             MyoSuite/myoSarcHandPoseFixed-v0
  myoSarcHandPoseRandom-v0            MyoSuite/myoSarcHandPoseRandom-v0
  myoSarcHandReachFixed-v0            MyoSuite/myoSarcHandReachFixed-v0
  myoSarcHandReachRandom-v0           MyoSuite/myoSarcHandReachRandom-v0
  myoSarcHandReorient100-v0           MyoSuite/myoSarcHandReorient100-v0
  myoSarcHandReorient8-v0             MyoSuite/myoSarcHandReorient8-v0
  myoSarcHandReorientID-v0            MyoSuite/myoSarcHandReorientID-v0
  myoSarcHandReorientOOD-v0           MyoSuite/myoSarcHandReorientOOD-v0
  myoSarcLegHillyTerrainWalk-v0       MyoSuite/myoSarcLegHillyTerrainWalk-v0
  myoSarcLegRoughTerrainWalk-v0       MyoSuite/myoSarcLegRoughTerrainWalk-v0
  myoSarcLegStairTerrainWalk-v0       MyoSuite/myoSarcLegStairTerrainWalk-v0
  myoSarcLegStandRandom-v0            MyoSuite/myoSarcLegStandRandom-v0
  myoSarcLegWalk-v0                   MyoSuite/myoSarcLegWalk-v0
  myoSarcTorsoExoPoseFixed-v0         MyoSuite/myoSarcTorsoExoPoseFixed-v0
  myoSarcTorsoPoseFixed-v0            MyoSuite/myoSarcTorsoPoseFixed-v0
  myoTorsoExoPoseFixed-v0             MyoSuite/myoTorsoExoPoseFixed-v0
  myoTorsoPoseFixed-v0                MyoSuite/myoTorsoPoseFixed-v0

The covered surface includes MyoBase reach, pose, key-turn, object-hold,
pen-twirl, reorient, walk, and terrain tasks; MyoChallenge tasks; MyoDM track
tasks; and the corresponding normal, sarcopenia, fatigue, and
reafferentation variants exposed by upstream.

Nine upstream IDs are still registered in EnvPool but are excluded from
official-oracle alignment tests because the pinned official package cannot
instantiate them under the MuJoCo 3.6 oracle environment:

::

  myoChallengeBimanual-v0
  myoSarcChallengeBimanual-v0
  myoFatiChallengeBimanual-v0
  myoChallengeSoccerP1-v0
  myoChallengeSoccerP2-v0
  myoSarcChallengeSoccerP1-v0
  myoSarcChallengeSoccerP2-v0
  myoFatiChallengeSoccerP1-v0
  myoFatiChallengeSoccerP2-v0


Render Compare
--------------

Reset and first-three-step render comparisons for every pinned official task
that the upstream oracle can instantiate: 389 tasks total, split into 151
MyoBase/Reorient/Walk/Terrain tasks, 48 MyoChallenge tasks, and 190 MyoDM
TrackEnv tasks. For each step pair, EnvPool is on the left and the pinned
MyoSuite renderer is on the right. The images are generated by
``third_party/myosuite/generate_render_sample.py`` from the pinned official
oracle and the same action sequence used by the render test. The render test
keeps a tight image-alignment gate so wrong cameras, models, and scenes fail,
while allowing a few backend-level pixel deltas. If EnvPool's public API
auto-resets a task within those three calls, the official oracle is reset at
that same reset boundary and synchronized only to the corresponding reset state.
The nine upstream Bimanual/Soccer IDs listed above remain registered but are
omitted from these official render sheets for the same oracle instantiation
failure.

MyoBase/Reorient/Walk/Terrain: 151 tasks.

.. image:: ../_static/render_samples/myosuite_myobase_official_compare.png
    :width: 900px
    :align: center

MyoChallenge: 48 tasks.

.. image:: ../_static/render_samples/myosuite_myochallenge_official_compare.png
    :width: 900px
    :align: center

MyoDM TrackEnv: 190 tasks.

.. image:: ../_static/render_samples/myosuite_myodm_official_compare.png
    :width: 900px
    :align: center
