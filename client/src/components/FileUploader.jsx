import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FiUpload } from 'react-icons/fi';
import axios from 'axios';

const FileUploader = () => {
  const [files, setFiles] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { getRootProps, getInputProps } = useDropzone({
    accept: 'application/pdf',
    onDrop: (acceptedFiles) => {
      setFiles([...files, ...acceptedFiles]);
    }
  });

  const handleSubmit = async () => {
    if (files.length === 0) {
      alert('No files to upload');
      return;
    }

    setIsSubmitting(true);

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await axios.post('/upload-endpoint', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      alert('Files uploaded successfully');
      setFiles([]);
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Failed to upload files');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="p-4">
      <div {...getRootProps({ className: 'dropzone' })} className="border-dashed border-2 border-gray-500 rounded-md p-6 text-center">
        <input {...getInputProps()} />
        <FiUpload className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-gray-500">Drag & drop PDF files here, or click to select files</p>
        <p className="mt-2 text-sm text-gray-400">You can upload PDF files with a limit of 750 pages per day</p>
      </div>
      <div className="mt-4">
        <h2 className="text-lg font-medium text-gray-200">Uploaded files</h2>
        <div className="mt-2 bg-gray-800 rounded-md p-4 text-center">
          {files.length === 0 ? (
            <p className="text-gray-500">No files uploaded</p>
          ) : (
            <ul className="list-disc list-inside">
              {files.map((file, index) => (
                <li key={index} className="text-gray-200">
                  {file.name}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <button
        onClick={handleSubmit}
        className={`mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 ${isSubmitting ? 'opacity-50 cursor-not-allowed' : ''}`}
        disabled={isSubmitting}
      >
        {isSubmitting ? 'Uploading...' : 'Submit'}
      </button>
    </div>
  );
};

export default FileUploader;
