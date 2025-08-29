using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using GLTFast;
using System.IO;

public class GLTFastLoader : MonoBehaviour
{
    public string filePath; // Path to the .glb or .gltf file
    public GameObject fpsCameraPrefab; // Prefab for the FPS camera

    async void Start()
    {
        var gltfImport = new GltfImport();
        bool success = await gltfImport.Load(new System.Uri(filePath).AbsoluteUri);
        if (success)
        {
            var instantiator = new GameObjectInstantiator(gltfImport, transform);
            success = await gltfImport.InstantiateMainSceneAsync(instantiator);
            if (success)
            {
                // Get the SceneInstance to access the instance's properties
                var sceneInstance = instantiator.SceneInstance;



                // Iterate through all instantiated objects and add colliders
                IterateObjects(transform);

                // Decrease lights' ranges
                if (sceneInstance.Lights != null)
                {
                    foreach (var glTFLight in sceneInstance.Lights)
                    {
                        glTFLight.range *= 0.0f;
                    }
                }

                // Setup the FPS controller
                SetupFPSController();
            }
            else
            {
                Debug.LogError("Failed to instantiate GLB scene.");
            }
        }
        else
        {
            Debug.LogError("Failed to load GLB file.");
        }
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
        // Skip objects with prefix "door_"
        if (obj.name.StartsWith("door_") | obj.name.StartsWith("room_volume_"))
        {
            return;
        }
        
        if (obj.GetComponent<MeshRenderer>() != null)
        {
            MeshCollider collider = obj.AddComponent<MeshCollider>();
            collider.convex = false;
        }
    }

    void SetupFPSController()
    {
        GameObject fpsCamera = Instantiate(fpsCameraPrefab, new Vector3(0, (float)2.5, 0), Quaternion.identity);
        fpsCamera.name = "FPSController";
    }
}

