# Hal-Eval: A Universal and Fine-grained Hallucination Evaluation Framework for Large Vision Language Models


<p align="center">
  🤗 <a href="https://huggingface.co/WisdomShell" target="_blank">Hugging Face</a>  • 🌐 <a href="http://se.pku.edu.cn/kcl/" target="_blank">PKU-KCL</a> •  🤖  <a href="http://27.188.73.160:7102/" target="_blank">Demo: Hal-Evaluator</a> 
 
</p>

## Introduction

Large Vision-Language Models (LVLMs) exhibit remarkable capabilities but struggle with "hallucinations"—inconsistencies between images and their descriptions. Previous hallucination evaluation studies on LVLMs have identified hallucinations in terms of objects, attributes, and relations but overlooked complex hallucinations that create an entire narrative around a fictional entity. In this paper, we introduce a refined taxonomy of hallucinations, featuring a new category: Event Hallucination. 
We then utilize advanced LLMs to generate and filter fine-grained hallucinatory data consisting of various types of hallucinations, with a particular focus on event hallucinations, laying the groundwork for integrating discriminative and generative evaluation methods within our universal evaluation framework. The proposed benchmark distinctively assesses LVLMs' ability to tackle a broad spectrum of hallucinations, making it a reliable and comprehensive tool for gauging LVLMs' efficacy in handling hallucinations. 

### Compared with Other Hallucination Benchmark

<p align="center">
<!DOCTYPE html>
<html>
<head>
<style>
  .green {
    color: green;
  }
</style>
</head>
<body>
<table>
  <tr>
    <th>Benchmark</th>
    <th colspan="2">Tasks</th>
    <th colspan="4">Discriminative Hallucination</th>
    <th colspan="4">Generative Hallucination</th>
  </tr>
  <tr>
    <th></th>
    <th>Dis</th>
    <th>Gen</th>
    <th>Object</th>
    <th>Attribute</th>
    <th>Relation</th>
    <th>Event</th>
    <th>Object</th>
    <th>Attribute</th>
    <th>Relation</th>
    <th>Event</th>
  </tr>
 <tr>
    <td>POPE</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
  <tr>
    <td>NOPE</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
  <tr>
    <td>CIEM</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
    <tr>
    <td>M-HalDetect</td>
     <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td>❌</td>
  </tr>
    <tr>
    <td>GAVIE</td>
    <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
    <tr>
    <td>FAITHScore</td>
        <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td>❌</td>
  </tr>
     <tr>
    <td>MMhal-Bench</td>
        <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>❌</td>
  </tr>
       <tr>
    <td>HaELM</td>
        <td>❌</td>
    <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>❌</td>
  </tr>
    <tr>
    <td>AMBER</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
     <td class="green">✔️</td>
    <td>❌</td>
     <td class="green">✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
      <tr>
    <td> <b>Hal-Eval</b> </td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
    <td class="green">✔️</td>
     <td class="green">✔️</td>
     <td class="green">✔️</td>
     <td class="green">✔️</td>
    <td class="green">✔️</td>
     <td class="green">✔️</td>
     <td class="green">✔️</td>
  </tr>
  <!-- Add more rows as needed -->
</table>
</body>
</html>
</p>


## Evaluation Dataset


## Discriminative Evalutation


## 