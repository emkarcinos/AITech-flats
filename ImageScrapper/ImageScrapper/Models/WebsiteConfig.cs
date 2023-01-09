using Newtonsoft.Json;

namespace ResellScrapperV3.Models;

public abstract class WebsiteConfig : IWebsiteConfig
{
    [JsonProperty("category")]
    public string Category { get; set; }
    [JsonProperty("gender")]
    public string Gender { get; set; }
    [JsonProperty("brand")]
    public string Brand { get; set; }
}