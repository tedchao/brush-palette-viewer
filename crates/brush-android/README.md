This runs brush using on Android.

To run this sample:
```
# One time setup:
export ANDROID_NDK_HOME="path/to/ndk"
export ANDROID_HOME="path/to/sdk"
rustup target add aarch64-linux-android
cargo install cargo-ndk

# Each time you change the rust code:
cargo ndk -t arm64-v8a -o app/src/main/jniLibs/  build
./gradlew build
./gradlew installDebug
adb shell am start -n com.splats.app/.MainActivity
```

You can also open this folder as a project in Android Studio and run things from there.
Nb: Running in Android Studio does _not_ rebuild the rust code.