using UnityEngine;



public class FPSController : MonoBehaviour
{
    public float speed = 5.0f;
    public float lookSpeed = 2.0f;

    private CharacterController characterController;
    private Camera playerCamera;
    private float rotationX = 0;

    void Start()
    {
        characterController = GetComponent<CharacterController>();
        playerCamera = Camera.main;
    }

    void Update()
    {
        MovePlayer();
        LookAround();
    }

    void MovePlayer()
    {
        float moveDirectionY = characterController.velocity.y;
        Vector3 move = transform.right * Input.GetAxis("Horizontal") + transform.forward * Input.GetAxis("Vertical");
        characterController.Move(move * speed * Time.deltaTime);
    }

    void LookAround()
    {
        rotationX -= Input.GetAxis("Mouse Y") * lookSpeed;
        rotationX = Mathf.Clamp(rotationX, -90, 90);
        playerCamera.transform.localRotation = Quaternion.Euler(rotationX, 0, 0);

        float rotationY = Input.GetAxis("Mouse X") * lookSpeed;
        transform.Rotate(0, rotationY, 0);
    }
}

