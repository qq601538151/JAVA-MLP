package com.ai.springai;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("chat")
public class ChatClientController {

    private ChatClient chatClient;

//    public ChatClientController(ChatClient.Builder chatClientBuilder) {
//        this.chatClient = chatClientBuilder.build();
//    }

    /**
     * 1.快速入门
     * @param userInput
     * @return
     */
    @GetMapping
    public String generation(String userInput) {
        return this.chatClient.prompt()
                .user(userInput)
                .call()
                .content();
    }

    /**
     * 2.返回ChatResponse
     * @param userInput
     * @return
     */
    @GetMapping("response")
    public String testResp(String userInput){
        ChatResponse chatResponse = chatClient.prompt()
                .user(userInput)
                .call()
                .chatResponse();
        return chatResponse.getResult().getOutput().getContent();
    }

    /**
     * 3.返回流式Response
     * @param userInput
     * @return
     */
    @GetMapping(value = "stream", produces = "text/plain;charset=UTF-8")
    public Flux<String> testStream(String userInput){
        Flux<String> output = chatClient.prompt()
                .user(userInput)
                .stream()
                .content();
        return output;
    }

    /**
     * 4.defaultSystem
     * @param openAiChatModel
     */
    public ChatClientController(OpenAiChatModel openAiChatModel) {
        this.chatClient = ChatClient
                .builder(openAiChatModel)
                .defaultSystem("记住，你是一个小学生，还是一个流氓，流氓要在言语中透露。PK是一个码农，也是一个帅哥！").build();
    }
}