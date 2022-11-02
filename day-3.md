# Day 3

YES! I managed to run a benchmark against the actual sagemaker endpoint deployment. The authentication was the tricky part...

I used a tool called [k6](https://k6.io/docs/). You express the test as a js file (`load_test.js`). For the authentication,
you have to set `AWS_ACCESSKEY`, `AWS_SECRETKEY` & `AWS_SECURITYTOKEN` env variables. They can be found in `~/.aws/credentials`
after you login with `aws-login`. Makes sure you use the right ones from the correct account. Then you can run the test using 

```
k6 run -i 2000  -u 10 load_test.js
```

`-i` flag is iteration. In our test one iteration is one request. `-u` are virtual users, basically concurrent requests. The above command will execute 2000 requests split between 10 virtual users.

When checking the results of the test, keep an eye on `http_req_failed`. If it is too high, investigate. I had 100% when I had problems with requests signing and it wasn't very obvious...

| model                   | all-mpnet-base-v2                    |  multi-qa-mpnet-base-dot-v1          |
|-------------------------|--------------------------------------|--------------------------------------|
| vanilla-ml.c5.la        | 15.988 rps, med=608.21, p(90)=644.06 | 14.966 rps, med=729.08, p(90)=768.71 |
| vanilla-ml.g4dn.xlarge  | 72.156 rps, med=121.92, p(90)=159.16 | 74.689 rps, med=115.71, p(90)=140.31 |

The `median` and `p(90)` latency is in ms.


Now to finally deploy the Triton server on sagemaker!
