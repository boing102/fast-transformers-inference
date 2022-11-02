// This is a k6 script that will load test the API

import http from 'k6/http';
import exec from 'k6/execution';
import aws4 from './aws4.js';


const payload = JSON.stringify({

  inputs: 'Simple Simple',
  
});

export default function () {

  const url = 'https://runtime.sagemaker.eu-central-1.amazonaws.com/endpoints/huggingface-pytorch-inference-2022-11-02-13-37-03-128/invocations';

  if (!__ENV.AWS_ACCESSKEY || !__ENV.AWS_SECRETKEY || !__ENV.AWS_SECURITYTOKEN) {
    exec.test.abort("Set the credentials as environment variables!");
  }

  let signature = aws4.sign({
    service: 'sagemaker',
    region: 'eu-central-1',
    method: 'POST',
    hostname: 'runtime.sagemaker.eu-central-1.amazonaws.com',
    path: '/endpoints/huggingface-pytorch-inference-2022-11-02-13-37-03-128/invocations',
    body: payload,
    headers: {
      'Content-Type': 'application/json',
      "User-Agent": "k6/0.40.0 (https://k6.io/)",
    },
  }, {
    accessKeyId: __ENV.AWS_ACCESSKEY,
    secretAccessKey: __ENV.AWS_SECRETKEY,
  });

  let final_headers = Object.assign(signature.headers,
    {
      'X-Amz-Security-Token': __ENV.AWS_SECURITYTOKEN,
    })
    
  const params = {
    headers: signature.headers,
  };

  let res = http.post(url, payload, params);
  
}
