package com.splats.app;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;

public class FilePicker {
    private static Activity _activity;
    public static final int REQUEST_CODE_PICK_FILE = 1;
    private static native void onFilePickerResult(byte[] data, String fileName);

    public static void Register(Activity activity) {
        _activity = activity;
    }

    public static void startFilePicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        Log.i("FilePicker", "GHello from Java!");

        _activity.startActivityForResult(intent, REQUEST_CODE_PICK_FILE);
    }

    public static void onActivityResult(int resultCode, Intent data) {
        // Nb: Need to ALWAYS call the native rust callback or it'll continue to wait forever.
        if (data == null || resultCode != Activity.RESULT_OK) {
            onFilePickerResult(null, "");
            return;
        }
        Uri uri = data.getData();
        if (uri == null) {
            onFilePickerResult(null, "");
            return;
        }
        String fileName = getFileName(uri);

        byte[] bytes = readBytesFromUri(uri);
        onFilePickerResult(bytes, fileName);
    }

    private static byte[] readBytesFromUri(Uri uri) {
        try {
            try (InputStream inputStream = _activity.getContentResolver().openInputStream(uri)) {
                return inputStream != null ? inputStream.readAllBytes() : null;
            }
        } catch (IOException _) {
            return null;
        }
    }

    private static String getFileName(Uri uri) {
        ContentResolver resolver = _activity.getContentResolver();
        Cursor returnCursor =
                resolver.query(uri, null, null, null, null);
        assert returnCursor != null;
        int nameIndex = returnCursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
        returnCursor.moveToFirst();
        String name = returnCursor.getString(nameIndex);
        returnCursor.close();
        return name;
    }
}
