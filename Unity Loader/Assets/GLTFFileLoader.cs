using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class GLTFFileLoader : MonoBehaviour
{
    public string filePath; // Path to the .glb or .gltf file
    public GameObject fpsCameraPrefab; // Prefab for the FPS camera

    void Start()
    {
        LoadGLBFile(filePath);
    }

    void LoadGLBFile(string path)
    {
        if (File.Exists(path))
        {
       	    
       	    /*
            byte[] fileData = File.ReadAllBytes(path);
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
            */
        }
        else
        {
            Debug.LogError("File not found: " + path);
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
            //AddCollider(child.gameObject);
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

