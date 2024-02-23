#include <iostream>
#include <cstdlib>
#include <ctime>

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

__global__ void merge(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary arrays
    int *L, *R;
    L = (int*)malloc(n1 * sizeof(int));
    R = (int*)malloc(n2 * sizeof(int));

    // Copy data to temporary arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // Merge the temporary arrays back into arr[left..right]
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    // Free temporary arrays
    free(L);
    free(R);
}

__global__ void mergeSort(int *arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        mergeSort<<<1,1>>>(arr, left, mid);
        mergeSort<<<1,1>>>(arr, mid + 1, right);

        // Merge the sorted halves
        merge<<<1,1>>>(arr, left, mid, right);
    }
}

void mergeSortCUDA(int *arr, int size) {
    int *d_arr;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_arr, size * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice));
    mergeSort<<<1,1>>>(d_arr, 0, size - 1);
    CUDA_CHECK_ERROR(cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_arr));
}

void printArray(int *arr, int size) {
    for (int i = 0; i < size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main() {
    const int size = 10;
    int arr[size];
    srand(time(NULL));
    for (int i = 0; i < size; i++)
        arr[i] = rand() % 100;

    std::cout << "Original array: ";
    printArray(arr, size);

    mergeSortCUDA(arr, size);

    std::cout << "Sorted array: ";
    printArray(arr, size);

    return 0;
}
