This paper delves into how language models represent time by testing a linear probe on the internal activations of the residual stream. 

In the graphs that I plotted, we find that there *is* a correlation between the real date of the data as what was predicted from the model, with a spearman coefficient of 0.6159 for layer 18, historical figures. There seems to be a concentration of data points in between 1500 and 2000, which I believe is a consequence of the data.

Furthermore, we plot how the model’s test_r2 changes throughout the layers: it increases steadily from layers 0-17, then plateaus. Both of these implementations match the original plots that Gurnee and Tegmark, except that with their model the test_r2 changes more at the halfway point. Their scatter plots also have a higher concentration of points towards the top right.

I used the Pythia410M model class to allow me to refine my plots / probes more rapidly, whereas they used the LLama 2 family of models.

One possible step that I can take is to reimplement the “Verification via Generalization” section of the assignment, as this section asks us to test whether the models are learning the true temporal representation or if they are relying on “membership features”. The paper cites this as an especially interesting result, as they find that they are able to generalize to unseen groups of held out data.