use anyhow::Result;
use async_channel::Sender;
use jni::objects::{GlobalRef, JByteArray, JClass, JStaticMethodID, JString};
use jni::signature::Primitive;
use jni::JNIEnv;
use lazy_static::lazy_static;
use std::sync::RwLock;

lazy_static! {
    static ref CHANNEL: RwLock<Option<Sender<Result<PickedFile>>>> = RwLock::new(None);
    static ref START_FILE_PICKER: RwLock<Option<JStaticMethodID>> = RwLock::new(None);
    static ref FILE_PICKER_CLASS: RwLock<Option<GlobalRef>> = RwLock::new(None);
}

pub fn cache_methods(env: &mut JNIEnv) {
    let class = env.find_class("com/splats/app/FilePicker").unwrap();
    let method = env
        .get_static_method_id(&class, "startFilePicker", "()V")
        .unwrap();
    *FILE_PICKER_CLASS.write().unwrap() = Some(env.new_global_ref(class).unwrap());
    *START_FILE_PICKER.write().unwrap() = Some(method);
}

#[derive(Clone, Debug)]
pub struct PickedFile {
    pub data: Vec<u8>,
    pub file_name: String,
}

pub async fn open_file_picker(java_vm: &jni::JavaVM) -> Result<PickedFile> {
    let (sender, receiver) = async_channel::bounded(1);
    {
        let mut channel = CHANNEL.write().unwrap();
        *channel = Some(sender);
    }

    let mut env = java_vm.attach_current_thread()?;
    let class = FILE_PICKER_CLASS.read().unwrap();
    let method = START_FILE_PICKER.read().unwrap();
    // SAFETY: This is safe as long as we cached the method in the right way, and
    // this matches the Java side. Not much more we can do here.
    let _ = unsafe {
        env.call_static_method_unchecked(
            class.as_ref().unwrap(),
            method.as_ref().unwrap(),
            jni::signature::ReturnType::Primitive(Primitive::Void),
            &[],
        )
    }?;
    receiver.recv().await?
}

#[no_mangle]
extern "system" fn Java_com_splats_app_FilePicker_onFilePickerResult<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    data: JByteArray<'local>,
    file_name: JString<'local>,
) {
    let picked_file = env.convert_byte_array(data).and_then(|data| {
        let file_name = env.get_string(&file_name)?.into();
        Ok(PickedFile { data, file_name })
    });
    let channel = CHANNEL.read().unwrap();
    channel
        .as_ref()
        .unwrap()
        .try_send(picked_file.map_err(|err| err.into()))
        .unwrap();
}
