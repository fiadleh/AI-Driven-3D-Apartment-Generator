using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System.IO;
using System.Collections.Generic;

public class GLTFWebLoader : MonoBehaviour
{
    public string fileUrl; // URL to the .glb or .gltf file
    public GameObject fpsCameraPrefab; // Prefab for the FPS camera

    void Start()
    {
        StartCoroutine(LoadGLBFile(fileUrl));
    }

    IEnumerator LoadGLBFile(string url)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(url))
        {
            // Request and wait for the desired file.
            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Failed to load GLB file: " + webRequest.error + fileUrl);
            }
            else
            {
                // Get the downloaded data.
                byte[] fileData = webRequest.downloadHandler.data;

                // Load the GLB file from the downloaded data.
                GameObject loadedScene = LoadGLBFromBytes(fileData);

                if (loadedScene != null)
                {
                    loadedScene.transform.position = Vector3.zero;
                    Debug.Log("GLB file loaded successfully.");
                    IterateObjects(loadedScene.transform);
                    SetupFPSController();
                }
                else
                {
                    Debug.LogError("Failed to parse GLB file.");
                }
            }
        }
    }

    GameObject LoadGLBFromBytes(byte[] bytes)
    {
        // For simplicity, this example assumes you are loading a very basic GLB file.
        // In a real implementation, you would need to handle the binary data parsing and mesh creation properly.
        GameObject rootObject = new GameObject("GLBRoot");
        
        // Parse GLB header
        if (bytes.Length < 12)
        {
            Debug.LogError("Invalid GLB file.");
            return null;
        }

        using (MemoryStream stream = new MemoryStream(bytes))
        using (BinaryReader reader = new BinaryReader(stream))
        {
            // Skip header (magic, version, length)
            reader.BaseStream.Seek(12, SeekOrigin.Begin);

            // Process chunks
            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                int chunkLength = reader.ReadInt32();
                string chunkType = new string(reader.ReadChars(4));
                byte[] chunkData = reader.ReadBytes(chunkLength);

                if (chunkType == "JSON")
                {
                    string json = System.Text.Encoding.UTF8.GetString(chunkData);
                    Debug.Log("GLB JSON: " + json);
                    // Here, you should parse the JSON and create the corresponding GameObjects and meshes.
                }
                else if (chunkType == "BIN\0")
                {
                    Debug.Log("GLB BIN chunk length: " + chunkLength);
                    // Here, you should use the binary data for buffer views, accessors, etc.
                }
            }
        }

        return rootObject;
    }

    void IterateObjects(Transform parent)
    {
        foreach (Transform child in parent)
        {
            AddCollider(child.gameObject);
            IterateObjects(child);
        }
    }

    void AddCollider(GameObject obj)
    {
        if (obj.GetComponent<MeshRenderer>() != null)
        {
            MeshCollider collider = obj.AddComponent<MeshCollider>();
            collider.convex = true;
        }
    }

    void SetupFPSController()
    {
        GameObject fpsCamera = Instantiate(fpsCameraPrefab, Vector3.zero, Quaternion.identity);
        fpsCamera.name = "FPSController";
    }
}

