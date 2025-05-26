package com.ai.springai;

import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("openai")
public class OpenAiController {

    @Autowired
    private OpenAiChatModel openAiChatModel;

    /**
     * 文本聊天
     * @param prompt
     * @return
     */
    @GetMapping("chat")
    public String chat(String prompt) {
        String result = this.openAiChatModel.call(prompt);
        return result;
    }

    /**
     * 文本聊天（流式输出）
     * @param prompt
     * @return
     */
    @GetMapping(value = "stream", produces = "text/plain;charset=UTF-8")
    public Flux<String> chatStream(String prompt) {
        Flux<String> resp = this.openAiChatModel.stream(prompt);
        return resp;
    }
    
    /**
     * 自定义运行参数
     * @param prompt
     * @return
     */
    @GetMapping("chat2")
    public String chat2(String prompt) {
        ChatResponse res = this.openAiChatModel.call(new Prompt(prompt, OpenAiChatOptions.builder()
                .withModel("gpt-3.5-turbo")
                .build()));
        String result = res.getResult().getOutput().getContent();
        return result;
    }

}